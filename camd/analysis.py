# Copyright Toyota Research Institute 2019

import abc
import warnings
import json
import pickle
import os
import numpy as np
import itertools
import pandas as pd
from camd import tqdm
from qmpy.analysis.thermodynamics.phase import Phase, PhaseData
from qmpy.analysis.thermodynamics.space import PhaseSpace
from multiprocessing import Pool, cpu_count
from pymatgen import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import (
    PhaseDiagram,
    PDPlotter,
    tet_coord,
    triangular_coord,
)
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen import Structure
from camd.utils.data import cache_matrio_data, \
    filter_dataframe_by_composition, ELEMENTS
from camd import CAMD_CACHE
from monty.os import cd
from monty.serialization import loadfn


class AnalyzerBase(abc.ABC):
    def __init__(self):
        self._initial_seed_indices = []

    @abc.abstractmethod
    def analyze(self, new_experimental_results, seed_data):
        """
        Analyze method, performs some operation on new
        experimental results in order to place them
        in the context of the seed data

        Args:
            new_experimental_results (DataFrame): new data
                to be added to the seed
            seed_data (DataFrame): current seed data from
                campaign

        Returns:
            (DataFrame): dataframe corresponding to the summary
                of the previous set of experiments
            (DataFrame): dataframe corresponding to the new
                seed data
        """

    @property
    def initial_seed_indices(self):
        """
        Returns: The list of indices contained in the initial seed. Intended to be set by the Campaign.
        """
        return self._initial_seed_indices


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

    def analyze(self, new_experimental_results, seed_data):
        new_seed = seed_data.append(new_experimental_results)
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
        return summary, new_seed


class AnalyzeStructures(AnalyzerBase):
    """
    This class tests if a list of structures are unique. Typically
    used for comparing hypothetical structures (post-DFT relaxation)
    and those from ICSD.
    """

    def __init__(self, structures=None, hull_distance=None):
        self.structures = structures if structures else []
        self.structure_ids = None
        self.unique_structures = None
        self.groups = None
        self.energies = None
        self.against_icsd = False
        self.structure_is_unique = None
        self.hull_distance = hull_distance
        super(AnalyzeStructures, self).__init__()

    def analyze(
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
                except:
                    warnings.warn("Unable to process structure {}".format(k))

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

    def analyze_vaspqmpy_jobs(self, jobs, against_icsd=False, use_energies=False):
        """
        Useful for analysis integrated as part of a campaign itself

        Args:
            jobs:
            against_icsd:

        Returns:
        """
        self.structure_ids = []
        self.structures = []
        self.energies = []
        for j, r in jobs.items():
            if r["status"] == "SUCCEEDED":
                self.structures.append(r["result"]["output"]["crystal"])
                self.structure_ids.append(j)
                self.energies.append(r["result"]["output"]["final_energy_per_atom"])
        if use_energies:
            return self.analyze(
                self.structures, self.structure_ids, against_icsd, self.energies
            )
        else:
            return self.analyze(self.structures, self.structure_ids, against_icsd)


class StabilityAnalyzer(AnalyzerBase):
    def __init__(self, hull_distance=0.05, parallel=cpu_count(), entire_space=False):
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
        """
        self.hull_distance = hull_distance
        self.parallel = parallel
        self.entire_space = entire_space
        self.space = None
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

    def analyze(self, new_experimental_results, seed_data):
        """
        Args:
            new_experimental_results (DataFrame): new experimental
                results to be added to the seed
            seed_data (DataFrame): seed to be augmented via
                the new_experimental_results

        Returns:
            (DataFrame): summary of the process, i. e. of
                the increment or experimental results
            (DataFrame): augmented seed data, i. e. "new"
                seed data according to the experimental results

        """
        # Aggregate seed_data and new experimental results
        new_seed = seed_data.append(new_experimental_results)
        include_columns = ["Composition", "delta_e"]
        # TODO: resolve this issue with drop_duplicates
        filtered = new_seed[include_columns].drop_duplicates(keep="last").dropna()

        if not self.entire_space:
            # Constrains the phase space to that of the target compounds.
            # More efficient when searching in a specified chemistry,
            # less efficient if larger spaces are without specified chemistry.
            total_comp = new_experimental_results["Composition"].dropna().sum()
            filtered = filter_dataframe_by_composition(filtered, total_comp)

        space = self.get_phase_space(filtered)
        new_phases = [p for p in space.phases if p.description in filtered.index]

        space.compute_stabilities(phases=new_phases, ncpus=self.parallel)

        # Compute new stabilities and update new seed, note that pandas will complain
        # if the index is not explicit due to multiple types (e. g. ints for OQMD
        # and strs for prototypes)
        new_data = pd.DataFrame(
            {"stability": [phase.stability for phase in new_phases]},
            index=[phase.description for phase in new_phases]
        )
        new_data["is_stable"] = new_data["stability"] <= self.hull_distance

        # TODO: This is implicitly adding "stability", and "is_stable" columns
        #       but could be handled more gracefully
        if "stability" not in new_seed.columns:
            new_seed = pd.concat([new_seed, new_data], axis=1, sort=False)
        else:
            new_seed.update(new_data)

        # Write hull figure to disk
        self.plot_hull(new_seed, new_experimental_results.index, filename="hull.png")

        # Compute summary metrics
        summary = self.get_summary(
            new_seed,
            new_experimental_results.index,
            initial_seed_indices=self.initial_seed_indices,
        )
        return summary, new_seed

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


        Returns:
            (DataFrame): dataframe summarizing processed
                experimental data including values for
                how many materials were discovered

        """
        # TODO: Right now analyzers don't know anything about the history
        #       of experiments, so can be difficult to determine marginal
        #       value of a given experimental run
        processed_new = new_seed.loc[new_ids]
        initial_seed_indices = initial_seed_indices if initial_seed_indices else []
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
        total_comp = df.loc[new_result_ids]["Composition"].dropna().sum()
        total_comp = Composition(total_comp)
        if len(total_comp) > 4:
            warnings.warn("Number of elements too high for phase diagram plotting")
            return None
        filtered = filter_dataframe_by_composition(df, total_comp)

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
        plotter = PDPlotter(pd, **plotkwargs)

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
                pd, **{"markersize": 0, "linestyle": "-", "linewidth": 2}
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

    def finalize(self, path="."):
        """
        Post-processing a dft campaign
        """
        update_run_w_structure(
            path, hull_distance=self.hull_distance, parallel=self.parallel
        )


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
        except:
            print(phase)
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


def update_run_w_structure(folder, hull_distance=0.2, parallel=True):
    """
    Updates a campaign grouped in directories with structure analysis

    """
    with cd(folder):
        required_files = ["seed_data.pickle"]
        if os.path.isfile("error.json"):
            error = loadfn("error.json")
            print("{} ERROR: {}".format(folder, error))

        if not all([os.path.isfile(fn) for fn in required_files]):
            print("{} ERROR: no seed data, no analysis to be done")
        else:
            iteration = -1
            jobs = {}
            while True:
                if os.path.isdir(str(iteration)):
                    jobs.update(
                        loadfn(os.path.join(str(iteration), "_exp_raw_results.json"))
                    )
                    iteration += 1
                else:
                    break
            with open("seed_data.pickle", "rb") as f:
                df = pickle.load(f)

            with open("experiment.pickle", "rb") as f:
                experiment = pickle.load(f)
            all_submitted, all_results = experiment.agg_history
            st_a = StabilityAnalyzer(hull_distance=hull_distance, parallel=parallel)
            summary, new_seed = st_a.analyze(df, pd.DataFrame())

            # Having calculated stabilities again, we plot the overall hull.
            st_a.plot_hull(
                new_seed,
                all_submitted.index,
                filename="hull_finalized.png",
                finalize=True,
            )

            stable_discovered = new_seed[new_seed["is_stable"]]
            s_a = AnalyzeStructures()
            s_a.analyze_vaspqmpy_jobs(jobs, against_icsd=True, use_energies=True)
            unique_s_dict = {}
            for i in range(len(s_a.structures)):
                if s_a.structure_is_unique[i] and (
                    s_a.structure_ids[i] in stable_discovered
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
