# Copyright Toyota Research Institute 2019
"""
Module for conducting analysis/postprocessing of
experiments and campaigns.  Contains the Analyzer
object, which performs this function.
"""

import abc
import warnings
import json
import os
import numpy as np
import itertools
import pandas as pd
from camd import tqdm
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from qmpy.analysis.thermodynamics.space import PhaseSpace
from multiprocessing import Pool, cpu_count
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import (
    PhaseDiagram,
    PDPlotter,
    tet_coord,
    triangular_coord,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from camd.utils.data import cache_matrio_data, filter_dataframe_by_composition, ELEMENTS
from camd import CAMD_CACHE


class AnalyzerBase(abc.ABC):
    """
    The AnalyzerBase class defines the contract
    for post-processing experiments and reporting
    on campaign state
    """

    def __init__(self):
        """
        Initialize an Analyzer.  Should contain all necessary
        state variables for consistent analysis methods
        """
        self._initial_seed_indices = []

    @abc.abstractmethod
    def analyze(self, campaign):
        """
        Analyze method, performs some operation on new
        experimental results in order to place them
        in the context of the seed data

        Args:
            campaign (Campaign): Campaign to be analyzed

        Returns:
            (DataFrame): dataframe corresponding to the analysis summary
        """


class GenericMaxAnalyzer(AnalyzerBase):
    """
    Generic analyzer that checks new data with a target column against a threshold to be crossed.
    """

    def __init__(self, threshold=0):
        """
        Args:
            threshold (int or float): The target values of new acquisitions are compared to find if they are above this
            threshold, to keep track of the performance in sequential framework.
        """
        self.threshold = threshold
        self.score = []
        self.best_examples = []
        super(GenericMaxAnalyzer, self).__init__()

    def analyze(self, campaign, finalize=False):
        """
        Analyzes the results of an experiment by finding
        the best examples and their scores

        Args:

        Returns:
            (pandas.DataFrame): one-row dataframe summarizing past results
            (pandas.DataFrame): new seed data to be passed to agent

        """
        new_seed = campaign.seed_data.append(campaign.experiment.get_results())
        self.score.append(np.sum(new_seed["target"] > self.threshold))
        self.best_examples.append(new_seed.loc[new_seed.target.idxmax()])
        new_discovery = (
            [self.score[-1] - self.score[-2]]
            if len(self.score) > 1
            else [self.score[-1]]
        )
        summary = pd.DataFrame(
            {
                "score": [self.score[-1]],
                "best_example": [self.best_examples[-1]],
                "new_discovery": new_discovery,
            }
        )
        return summary


# TODO: this isn't yet a true analyzer - still trying to figure out
#   how to implement this consistently
class AnalyzeStructures(AnalyzerBase):
    """
    This class tests if a list of structures are unique. Typically
    used for comparing hypothetical structures (post-DFT relaxation)
    and those from ICSD.
    """

    def __init__(self, structures=None, hull_distance=None):
        """
        Analyzer for structural analysis of jobs

        Args:
            structures ([Structure]): list of a-priori structures to
                compare against
            hull_distance ([float]): hull_distance by which to filter
                results

        """
        self.structures = structures if structures else []
        self.structure_ids = None
        self.unique_structures = None
        self.groups = None
        self.energies = None
        self.against_icsd = False
        self.structure_is_unique = None
        self.hull_distance = hull_distance
        super(AnalyzeStructures, self).__init__()

    def _analyze_structures(
        self, structures=None, structure_ids=None, against_icsd=False, energies=None
    ):
        """
        One encounter of a given structure will be labeled as True, its
        remaining matching structures as False.

        Args:
            structures (list): a list of structures to be compared.
            structure_ids (list): uids of structures, optional.
            against_icsd (bool): whether a comparison to icsd is also made.
            energies (list): list of energies (per atom) corresponding
                to structures. If given, the lowest energy instance of a
                given structure will be return as the unique one. Otherwise,
                there is no such guarantee. (optional)

        Returns:
            ([bool]) list of bools corresponding to the given list of
                structures corresponding to uniqueness
        """
        self.structures = structures
        self.structure_ids = structure_ids
        self.against_icsd = against_icsd
        self.energies = energies

        smatch = StructureMatcher()
        self.groups = smatch.group_structures(structures)
        self.structure_is_unique = []

        if self.energies:
            for i in range(len(self.groups)):
                self.groups[i] = [
                    x
                    for _, x in sorted(
                        zip(
                            [
                                self.energies[self.structures.index(s)]
                                for s in self.groups[i]
                            ],
                            self.groups[i],
                        )
                    )
                ]

        self._unique_structures = [i[0] for i in self.groups]
        for s in structures:
            if s in self._unique_structures:
                self.structure_is_unique.append(True)
            else:
                self.structure_is_unique.append(False)
        self._not_duplicate = self.structure_is_unique

        if self.against_icsd:
            structure_file = "oqmd1.2_exp_based_entries_structures.json"
            cache_matrio_data(structure_file)
            with open(os.path.join(CAMD_CACHE, structure_file), "r") as f:
                icsd_structures = json.load(f)
            chemsys = set()
            for s in self._unique_structures:
                chemsys = chemsys.union(set(s.composition.as_dict().keys()))

            self.icsd_structs_inchemsys = []
            for k, v in icsd_structures.items():
                try:
                    s = Structure.from_dict(v)
                    elems = set(s.composition.as_dict().keys())
                    if elems == chemsys:
                        self.icsd_structs_inchemsys.append(s)
                # TODO: can we make this exception more specific,
                #  do we have an example where this fails?
                except Exception as e:
                    warnings.warn("Unable to process structure {}".format(k))
                    warnings.warn("Error: {}".format(e))

            self.matching_icsd_strs = []
            for i in range(len(structures)):
                if self.structure_is_unique[i]:
                    match = None
                    for s2 in self.icsd_structs_inchemsys:
                        if smatch.fit(self.structures[i], s2):
                            match = s2
                            break
                    self.matching_icsd_strs.append(
                        match
                    )  # store the matching ICSD structures.
                else:
                    self.matching_icsd_strs.append(None)

            # Flip matching bools, and create a filter
            self._icsd_filter = [not i for i in self.matching_icsd_strs]
            self.structure_is_unique = (
                np.array(self.structure_is_unique) * np.array(self._icsd_filter)
            ).tolist()
            self.unique_structures = list(
                itertools.compress(self.structures, self.structure_is_unique)
            )
        else:
            self.unique_structures = self._unique_structures

        # We store the final list of unique structures as unique_structures.
        # We return a corresponding list of bool to the initial structure
        # list provided.
        return self.structure_is_unique

    def analyze(self, campaign):
        """
        Analysis method for structures, not yet implemented

        Args:
            campaign (Campaign): campaign object to be analyzed

        Returns:
            (pd.DataFrame): summary of results from structure analysis

        """
        raise NotImplementedError

    def analyze_vaspqmpy_jobs(self, jobs, against_icsd=False, use_energies=False):
        """
        Useful for analysis integrated as part of a campaign itself

        Args:
            jobs (pd.DataFrame): dataframe of DFT experiment results
            against_icsd (bool): whether to validate against ICSD or not

        Returns:
        """
        self.structure_ids = []
        self.structures = []
        self.energies = []
        for j, r in jobs.iterrows():
            if r["status"] == "SUCCEEDED":
                # This is a switch for OQMD vs. MP
                if "output" in r:
                    final_structure = r["output"]["structure"]
                    self.structures.append(final_structure)
                    self.energies.append(r["output"]["energy_per_atom"])
                    self.structure_ids.append(j)
                else:
                    final_structure = r["result"].final_structure
                    self.structures.append(final_structure)
                    self.structure_ids.append(j)
                    self.energies.append(
                        r["result"].final_energy / len(final_structure)
                    )
        if use_energies:
            return self._analyze_structures(
                self.structures, self.structure_ids, against_icsd, self.energies
            )
        else:
            return self._analyze_structures(
                self.structures, self.structure_ids, against_icsd
            )


class StabilityAnalyzer(AnalyzerBase):
    """
    Analyzer object for stability campaigns
    """

    def __init__(
        self,
        hull_distance=0.05,
        parallel=cpu_count(),
        entire_space=False,
        plot=True,
        initial_seed_indices=None,
    ):
        """
        The Stability Analyzer is intended to analyze DFT-result
        data in the context of a global compositional seed in
        order to determine phase stability.

        Args:
            hull_distance (float): distance above hull below
                which to deem a material "stable"
            parallel (bool): flag for whether or not
                multiprocessing is to be used
            # TODO: is there ever a case where you would
            #       would want to do the entire space?
            entire_space (bool): flag for whether to analyze
                entire space of results or just new chemical
                space
            plot (bool): whether to generate plot as part of
                standard analyze sequence
        """
        self.hull_distance = hull_distance
        self.parallel = parallel
        self.entire_space = entire_space
        self.space = None
        self.plot = plot
        self.initial_seed_indices = initial_seed_indices
        super(StabilityAnalyzer, self).__init__()

    @staticmethod
    def get_phase_space(dataframe):
        """
        Gets PhaseSpace object associated with dataframe

        Args:
            dataframe (DataFrame): dataframe with columns "Composition"
                containing formula and "delta_e" containing
                formation energy per atom
        """
        phases = []
        for data in dataframe.iterrows():
            phases.append(
                Phase(
                    data[1]["Composition"],
                    energy=data[1]["delta_e"],
                    per_atom=True,
                    description=data[0],
                )
            )
        for el in ELEMENTS:
            phases.append(Phase(el, 0.0, per_atom=True))

        pd = PhaseData()
        pd.add_phases(phases)
        space = PhaseSpaceAL(bounds=ELEMENTS, data=pd)
        return space

    @staticmethod
    def add_stability(df, hull_distance=0.2, parallel=cpu_count()):
        """
        Adds stability to a dataframe

        Args:
            df (pd.DataFrame): dataframe of entries with `delta_e` field
            hull_distance (float): distance to the hull to classify as stable
            parallel (int): number of parallel processes to use

        Returns:
            (pd.DataFrame): dataframe of stabilities and is_stable

        """
        space = StabilityAnalyzer.get_phase_space(df)
        new_phases = [p for p in space.phases if p.description in df.index]

        space.compute_stabilities(phases=new_phases, ncpus=parallel)

        # Compute new stabilities and update new seed, note that pandas will complain
        # if the index is not explicit due to multiple types (e. g. ints for OQMD
        # and strs for prototypes)
        new_data = pd.DataFrame(
            {"stability": [phase.stability for phase in new_phases]},
            index=[phase.description for phase in new_phases],
        )
        new_data["is_stable"] = new_data["stability"] <= hull_distance
        return new_data

    def analyze(self, campaign, finalize=False):
        """
        Args:
            campaign (Campaign): CAMD campaign
            finalize (bool): whether or not analysis is final

        Returns:
            (DataFrame): summary of the process, i. e. of
                the increment or experimental results

        """
        # On first run get initial seed indices
        if self.initial_seed_indices is None:
            self.initial_seed_indices = campaign.seed_data.index
        # Check for new results
        new_experimental_results = campaign.experiment.get_results()
        new_comp = new_experimental_results["Composition"].sum()
        new_experimental_results = new_experimental_results.dropna(subset=["delta_e"])
        if not finalize:
            new_seed = campaign.seed_data.append(new_experimental_results)
        else:
            new_seed = campaign.seed_data

        # Aggregate seed_data and new experimental results
        include_columns = ["Composition", "delta_e"]
        filtered = new_seed[include_columns].drop_duplicates(keep="last").dropna()

        if not self.entire_space:
            # Constrains the phase space to that of the target compounds.
            # More efficient when searching in a specified chemistry,
            # less efficient if larger spaces are without specified chemistry.
            filtered = filter_dataframe_by_composition(filtered, new_comp)

        new_data = self.add_stability(
            filtered, hull_distance=self.hull_distance, parallel=self.parallel
        )

        # # TODO: This is implicitly adding "stability", and "is_stable" columns
        #       but could be handled more gracefully
        if "stability" not in new_seed.columns:
            new_seed = pd.concat([new_seed, new_data], axis=1, sort=False)
        else:
            new_seed.update(new_data)

        # Write hull figure to disk
        if self.plot:
            self.plot_hull(
                filtered, new_experimental_results.index, filename="hull.png"
            )

        # Compute summary metrics
        summary = self.get_summary(
            new_seed,
            new_experimental_results.index,
            initial_seed_indices=self.initial_seed_indices,
        )
        return summary

    @staticmethod
    def get_summary(new_seed, new_ids, initial_seed_indices=None):
        """
        Gets summary row for given experimental results after
        preliminary stability analysis.  This is not meant
        to provide the basis for a generic summary method
        and is particular to the StabilityAnalyzer.

        Args:
            new_seed (DataFrame): dataframe corresponding to
                new processed seed
            new_ids ([]): list of index values for those
                experiments that are "new"
            initial_seed_indices ([]): indices of the initial
                seed

        Returns:
            (DataFrame): dataframe summarizing processed
                experimental data including values for
                how many materials were discovered

        """
        # TODO: Right now analyzers don't know anything about the history
        #       of experiments, so can be difficult to determine marginal
        #       value of a given experimental run
        processed_new = new_seed.loc[new_ids]
        initial_seed_indices = (
            initial_seed_indices if initial_seed_indices is not None else []
        )
        total_discovery = new_seed.loc[
            ~new_seed.index.isin(initial_seed_indices)
        ].is_stable.sum()
        return pd.DataFrame(
            {
                "new_candidates": [len(processed_new)],
                "new_discovery": [processed_new.is_stable.sum()],
                "total_discovery": [total_discovery],
            }
        )

    def plot_hull(self, df, new_result_ids, filename=None, finalize=False):
        """
        Generate plots of convex hulls for each of the runs

        Args:
            df (DataFrame): dataframe with formation energies and formulas
            new_result_ids ([]): list of new result ids (i. e. indexes
                in the updated dataframe)
            filename (str): filename to output, if None, no file output
                is produced
            finalize (bool): flag indicating whether to include all new results

        Returns:
            (pyplot): plotter instance
        """
        # Generate all entries
        total_comp = Composition(df["Composition"].sum())
        if len(total_comp) > 4:
            warnings.warn("Number of elements too high for phase diagram plotting")
            return None
        filtered = filter_dataframe_by_composition(df, total_comp)
        filtered = filtered[["delta_e", "Composition"]]
        filtered = filtered.dropna()

        # Create computed entry column with un-normalized energies
        filtered["entry"] = [
            ComputedEntry(
                Composition(row["Composition"]),
                row["delta_e"] * Composition(row["Composition"]).num_atoms,
                entry_id=index,
            )
            for index, row in filtered.iterrows()
        ]

        ids_prior_to_run = list(set(filtered.index) - set(new_result_ids))
        if not ids_prior_to_run:
            warnings.warn("No prior data, prior phase diagram cannot be constructed")
            return None

        # Create phase diagram based on everything prior to current run
        entries = filtered.loc[ids_prior_to_run]["entry"].dropna()

        # Filter for nans by checking if it's a computed entry
        pg_elements = sorted(total_comp.keys())
        pd = PhaseDiagram(entries, elements=pg_elements)
        plotkwargs = {
            "markerfacecolor": "white",
            "markersize": 7,
            "linewidth": 2,
        }
        if finalize:
            plotkwargs.update({"linestyle": "--"})
        else:
            plotkwargs.update({"linestyle": "-"})
        plotter = PDPlotter(pd, backend="matplotlib", **plotkwargs)

        getplotkwargs = {"label_stable": False} if finalize else {}
        plot = plotter.get_plot(**getplotkwargs)

        # Get valid results
        valid_results = [
            new_result_id
            for new_result_id in new_result_ids
            if new_result_id in filtered.index
        ]

        if finalize:
            # If finalize, we'll reset pd to all entries at this point to
            # measure stabilities wrt. the ultimate hull.
            pd = PhaseDiagram(filtered["entry"].values, elements=pg_elements)
            plotter = PDPlotter(
                pd,
                backend="matplotlib",
                **{"markersize": 0, "linestyle": "-", "linewidth": 2}
            )
            plot = plotter.get_plot(plt=plot)

        for entry in filtered["entry"][valid_results]:
            decomp, e_hull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
            if e_hull < self.hull_distance:
                color = "g"
                marker = "o"
                markeredgewidth = 1
            else:
                color = "r"
                marker = "x"
                markeredgewidth = 1

            # Get coords
            coords = [entry.composition.get_atomic_fraction(el) for el in pd.elements][
                1:
            ]
            if pd.dim == 2:
                coords = coords + [pd.get_form_energy_per_atom(entry)]
            if pd.dim == 3:
                coords = triangular_coord(coords)
            elif pd.dim == 4:
                coords = tet_coord(coords)
            plot.plot(
                *coords,
                marker=marker,
                markeredgecolor=color,
                markerfacecolor="None",
                markersize=11,
                markeredgewidth=markeredgewidth
            )

        if filename is not None:
            plot.savefig(filename, dpi=70)
        plot.close()

    def finalize(self, campaign):
        """
        Post-processing a dft campaign
        """
        # self.analyze(campaign, finalize=True)
        all_submitted, all_results = campaign.experiment.agg_history
        old_results = campaign.seed_data.drop(all_results.index, errors="ignore")
        new_results = campaign.seed_data.drop(old_results.index)
        st_a = StabilityAnalyzer(
            hull_distance=self.hull_distance,
            parallel=self.parallel,
            entire_space=False,
            plot=False,
        )

        # Having calculated stabilities again, we plot the overall hull.
        # Filter by chemsys
        new_comp = new_results["Composition"].sum()
        filtered = filter_dataframe_by_composition(campaign.seed_data, new_comp)
        filtered = filtered.dropna(subset=["delta_e"])

        st_a.plot_hull(
            filtered,
            all_submitted.index,
            filename="hull_finalized.png",
            finalize=True,
        )

        stabs = self.add_stability(
            filtered, hull_distance=self.hull_distance, parallel=self.parallel
        )
        stable_discovered = stabs[stabs["is_stable"].fillna(False)]

        # Analyze structures if present in experiment
        if "structure" in all_results.columns:
            s_a = AnalyzeStructures()
            s_a.analyze_vaspqmpy_jobs(all_results, against_icsd=True, use_energies=True)
            unique_s_dict = {}
            for i in range(len(s_a.structures)):
                if s_a.structure_is_unique[i] and (
                    s_a.structure_ids[i] in stable_discovered.index
                ):
                    unique_s_dict[s_a.structure_ids[i]] = s_a.structures[i]

            with open("discovered_unique_structures.json", "w") as f:
                json.dump(dict([(k, s.as_dict()) for k, s in unique_s_dict.items()]), f)

            with open("structure_report.log", "w") as f:
                f.write("consumed discovery unique_discovery duplicate in_icsd \n")
                f.write(
                    str(len(all_submitted))
                    + " "
                    + str(len(stable_discovered))
                    + " "
                    + str(len(unique_s_dict))
                    + " "
                    + str(len(s_a.structures) - sum(s_a._not_duplicate))
                    + " "
                    + str(sum([not i for i in s_a._icsd_filter]))
                )
        return True


class PhaseSpaceAL(PhaseSpace):
    """
    Modified qmpy.PhaseSpace for GCLP based stability computations
    TODO: basic multithread or Gurobi for gclp
    """

    def compute_stabilities(self, phases, ncpus=cpu_count()):
        """
        Calculate the stability for every Phase.

        Args:
            phases ([Phase]): list of Phases for which to compute
                stability
            ncpus (int): number of cpus to use, i. e. processes
                to use

        Returns:
            ([float]) stability values for all phases
        """
        self.update_phase_dict(ncpus=ncpus)
        if ncpus > 1:
            with Pool(ncpus) as pool:
                stabilities = pool.map(self.compute_stability, phases)
            # Pool doesn't always modify the phases directly,
            # so assign stability after
            for phase, stability in zip(phases, stabilities):
                phase.stability = stability
        else:
            stabilities = [self.compute_stability(phase) for phase in tqdm(phases)]

        return stabilities

    def compute_stability(self, phase):
        """
        Computes stability for a given phase in the phase
        diagram

        Args:
            phase (Phase): phase for which to compute
                stability

        Returns:
            (float): stability of given phase

        """
        # If the phase name (formula) is in the set of minimal
        # phases by formula, compute it relative to that minimal phase
        if phase.name in self.phase_dict:
            phase.stability = (
                phase.energy
                - self.phase_dict[phase.name].energy
                + self.phase_dict[phase.name].stability
            )
        else:
            phase.stability = self._compute_stability_gclp(phase)

        return phase.stability

    def _compute_stability_gclp(self, phase):
        """
        Computes stability using gclp.  The function
        is still a little unstable, so we use a blank
        try-except to let it do what it can.

        Args:
            phase (Phase): phase for which to compute
                stability using gclp

        Returns:
            (float): stability

        """
        try:
            phase.stability = phase.energy - self.gclp(phase.unit_comp)[0]
        # TODO: do we have an example where this fails?  Can we provide
        #  a more concrete exception?
        except Exception as e:
            print(phase, "stability determination failed, error {}".format(e))
            phase.stability = np.nan
        return phase.stability

    def update_phase_dict(self, ncpus=cpu_count()):
        """
        Function to update the phase dict associated with
        the PhaseSpaceAL

        Args:
            ncpus (int): number of processes to use

        Returns:
            (None)

        """
        uncomputed_phases = [
            phase for phase in self.phase_dict.values() if phase.stability is None
        ]
        if ncpus > 1:
            # Compute stabilities, then update, pool doesn't modify attribute
            with Pool(ncpus) as pool:
                stabilities = pool.map(self._compute_stability_gclp, uncomputed_phases)
            for phase, stability in zip(uncomputed_phases, stabilities):
                phase.stability = stability
        else:
            for phase in uncomputed_phases:
                self._compute_stability_gclp(phase)
        assert (
            len(
                [phase for phase in self.phase_dict.values() if phase.stability is None]
            )
            == 0
        )


class GenericATFAnalyzer(AnalyzerBase):
    """
    Generic analyzer provide AL metrics analysis, including:

    deALM: decision efficiency metric
    anyALM: any discover from top n percentile catalyst
    allALM: fraction of top m percentile
    simALM: similarity between seed_df and new candidate

    This is only used in simulated discovery campaign,
    aggregated selection of candidates from one specific agent
    should be passed in
    """

    def __init__(self, percentile=0.01):
        """
        Notice the default of SL is to maximize a value, if the target value is
        overpotential, please remember to negate the target values

        Args:
            percentile (float):
                top percentile of candidates considered as "good"
        """
        self.percentile = percentile
        super(GenericATFAnalyzer, self).__init__()

    def analyze(self, campaign, finalize=False):
        """
        run analysis one by one

        Args:
            campaign (Campaign): campaign object to analyze
            finalize (bool): whether this analysis is the final one

        Returns:
            (DataFrame): dataframe with summary stats from campaign

        """
        seed_data = campaign.seed_data
        new_experimental_results = campaign.experiment.get_results()
        summary = pd.DataFrame()
        unsampled_candidates = campaign.experiment.dataframe.loc[campaign.candidate_data.index]
        all_results = unsampled_candidates.append(campaign.experiment.dataframe.loc[campaign.seed_data.index])
        unsampled_candidates = unsampled_candidates.drop(new_experimental_results.index)
        de_alms = self.get_de_alm(
            new_experimental_results,
            unsampled_candidates
        )
        summary["deALM"] = [de_alms]

        any_alm = self.get_any_alm(
            new_experimental_results,
            all_results,
            percentile=self.percentile,
        )
        summary["anyALM"] = [any_alm]

        all_sampled = seed_data.append(new_experimental_results).drop_duplicates()
        all_alm = self.get_all_alm(
            all_sampled,
            all_results,
            percentile=self.percentile,
        )
        summary["allALM"] = [all_alm]
        # simALM_val = self.gather_simALM(
        #     new_experimental_results.copy(), seed_data.copy()
        # )
        # summary["simALM"] = [simALM_val]

        return summary

    @staticmethod
    def get_de_alm(samples, unsampled_results):
        """
        Compute the decision efficiency ALM for one agent.

        de_ALM reflects the percentile of a sample relative
        the remaining unsampled results, and is computed as
        2*f_i - 1 in order to adopt a range from [-1, 1] which
        can be easily compared to the random case (for which
        de_ALM should be 0.)

        If the latest sample was the "best" choice, de_ALM is 1,
        if it was the worst, de_ALM is -1.

        Args:
            samples (pd.DataFrame): latest samples
            unsampled_results (pd.DataFrame): not yet sampled results
                from ATF simulation

        Returns:
            (pd.DataFrame): de_ALM
        """
        # go through each selected candidate, find the rank
        de_ALMs = []
        for idx, sample in samples.iterrows():
            temp = unsampled_results.append(sample)
            sample_rank_pct = temp['target'].rank(pct=True).loc[idx]
            de_ALMs.append(2 * sample_rank_pct - 1)

        return de_ALMs

    @staticmethod
    def get_any_alm(
        samples, all_results, percentile
    ):
        """
        Compute the any ALM for one agent, which is a boolean
        describing whether or not some sample from the samples thus
        far is present in the top n percentile of materials, e.g.
        if some sample in samples is in the top 1% of all results
        returns True.  Note that the any_alm from Rohr et al. is
        the first iteration in which this is true for a given
        random initialization of an SL campaign.

        Args:
            samples (pd.DataFrame):
                selection of candidiates in the right order
            all_results (pd.DataFrame):
                all_results to use for percentile determination
            percentile (float):
                top percentile of candidates considered as "good"

        Returns:
            (bool): True if any of the samples are in the top
                percentile of all_results, False otherwise
        """
        sample_ranks = all_results['target'].rank(pct=True).loc[samples.index]
        return (sample_ranks > 1 - percentile).any()

    @staticmethod
    def get_all_alm(
        all_samples, all_possible_samples, percentile
    ):
        """
        Compute the all ALM for one agent, defined as the fraction of
        samples above the target percentile which are contained in
        the samples taken up to this point. For example, if 50% of
        the possible catalysts in the top 1% overpotential have
        been sampled, allALM is 0.5.

        Args:
            all_samples (pd.DataFrame): all taken samples up to this point
            all_possible_samples (pd.DataFrame): all possible samples to
                compute percentiles from
            percentile (float):
                top percentile of candidates considered as "good"
        Returns:
            (float): percentage of top percentile candidates contained
                in aggregated sample set up to this point
        """
        # sort df using target values
        ranked = all_possible_samples['target'].rank(pct=True)
        top_percentile = ranked[ranked > 1 - percentile]
        all_alm = len(top_percentile.index.intersection(all_samples.index)) / len(top_percentile)
        return all_alm

    # joseph.montoya: I don't know what this is, so I'm going to comment it for now
    #   it looks like it might be a metric related to whether the most recent samples are
    #   inside or outside the domain?
    # def gather_simALM(self, new_experimental_results, seed_data):
    #     """
    #     compute the sim ALM for one agent
    #     Args:
    #         new_experimental_results (pd.DataFrame):
    #             selection of candidiates in the right order
    #         seed_data (pd.DataFrame):
    #             seed data used before first-round CAMD selection starts
    #     Returns:
    #         simALM (np.array):
    #             sim ALM for one specific agent
    #     """
    #     simALM = []
    #     # drop the target values so that only features are left
    #     seed_data.drop("target", inplace=True, axis=1)
    #     new_experimental_results.drop("target", inplace=True, axis=1)
    #     new_experimental_results.reset_index(inplace=True, drop=True)
    #     # go through the new candidate one by one
    #     for i in range(new_experimental_results.shape[0]):
    #         decision_fecture_vector = np.array(new_experimental_results.loc[i].tolist())
    #         seed_feature_vector = np.array(seed_data.values.tolist())
    #         similarity = (
    #             np.linalg.norm(seed_feature_vector - decision_fecture_vector)
    #             / seed_data.shape[0]
    #         )
    #         simALM.append(similarity)
    #         seed_data = pd.concat(
    #             [seed_data, new_experimental_results[i: (i + 1)]], axis=0
    #         )
    #     simALM = np.array(simALM)
    #     return simALM


class MultiAnalyzer(AnalyzerBase):
    """
    The multi-fidelity analyzer.
    """

    def __init__(
        self,
        target_prop,
        prop_range,
        total_expt_queried=0,
        total_expt_discovery=0,
        analyze_cost=False,
        total_cost=0.0,
    ):
        """
        Args:
            target_prop (str)        The name of the target property, e.g. "bandgap".
            prop_range (list)        The range of the target property that is considered
                                     ideal.
            total_expt_queried (int) The total experimental queries after nth iteration.
            tot_expt_discovery (int) The total experimental discovery after nth iteration.
            analyze_cost (bool)      If the input has "cost" column, also analyze that information.
            total_cost(float)        The total cost of the hypotheses after nth iteration.
        """
        self.target_prop = target_prop
        self.prop_range = prop_range
        self.total_expt_queried = total_expt_queried
        self.total_expt_discovery = total_expt_discovery
        self.analyze_cost = analyze_cost
        self.total_cost = total_cost

    def _filter_df_by_prop_range(self, df):
        """
        Helper function that filters df by property range

        Args:
            df   A pd.Dataframe to be filtered.
        """
        return df.loc[
            (df[self.target_prop] >= self.prop_range[0])
            & (df[self.target_prop] <= self.prop_range[1])
        ]

    def analyze(self, campaign, finalize=False):
        """
        Analysis method for campaign

        Args:
            campaign (Campaign):
            finalize (bool):

        Returns:
            (DataFrame): pandas dataframe with summary of the results

        """
        return self._analyze_results(
            campaign.experiment.get_results(), campaign.seed_data
        )

    def _analyze_results(self, new_experimental_results, seed_data):
        """
        Analyze results of multi-fidelity data

        Args:
            new_experimental_results (pd.DataFrame):
            seed_data (pd.DataFrame):

        Returns:
            (pd.DataFrame): summary of last iteration
            (pd.DataFrame): new seed data

        """
        new_expt_hypotheses = new_experimental_results.loc[
            new_experimental_results["expt_data"] == 1
        ]
        new_discoveries = self._filter_df_by_prop_range(new_expt_hypotheses)

        # total discovery = up to (& including) the current iteration
        self.total_expt_queried += len(new_expt_hypotheses)
        self.total_expt_discovery += len(new_discoveries)
        if self.total_expt_queried != 0:
            success_rate = self.total_expt_discovery / self.total_expt_queried
        else:
            success_rate = 0

        summary = pd.DataFrame(
            {
                "expt_queried": [len(new_expt_hypotheses)],
                "total_expt_queried": [self.total_expt_queried],
                "new_discovery": [len(new_discoveries)],
                "total_expt_discovery": [self.total_expt_discovery],
                "total_regret": [self.total_expt_queried - self.total_expt_discovery],
                "success_rate": [success_rate],
            }
        )

        if self.analyze_cost:
            iter_cost = np.sum(new_experimental_results["cost"])
            self.total_cost += iter_cost
            summary["iteration_cost"] = [iter_cost]
            summary["total_cost"] = [self.total_cost]
            if self.total_expt_discovery != 0:
                average_cost_per_discovery = self.total_cost / self.total_expt_discovery
            else:
                average_cost_per_discovery = np.nan
            summary["average_cost_per_discovery"] = [average_cost_per_discovery]
        return summary
