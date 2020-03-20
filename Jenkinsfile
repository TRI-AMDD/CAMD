// define a tag according to the Docker tag rules https://docs.docker.com/engine/reference/commandline/tag/
// the hash sign (#) is problematic when using it in bash, instead of working around this problem, just replace all
// punctuation with dash (-)
def dockerTagWithoutBuildNumber = "internal-${env.BRANCH_NAME}".toLowerCase().replaceAll("\\p{Punct}", "-").replaceAll("\\p{Space}", "-")
def dockerTag = "${dockerTagWithoutBuildNumber}-${env.BUILD_NUMBER}"
def dockerTagLatest = "${dockerTagWithoutBuildNumber}-latest"
def awsRegion = "us-west-2"
def githubOrg = "materials"
def dockerRegistry = "251589461219.dkr.ecr.${awsRegion}.amazonaws.com"
def dockerRegistryPrefix = "camd-worker"
def dockerfile = "Dockerfile"
def buildLink = "<${env.BUILD_URL}|${env.JOB_NAME} ${env.BUILD_NUMBER}>"
// Define the python test, coverage, lint procedures

node {
  timestamps {

    ansiColor('xterm') {
      try {
        properties properties: [
          [$class: 'BuildDiscarderProperty', strategy: [$class: 'LogRotator', artifactDaysToKeepStr: '100', artifactNumToKeepStr: '1000', daysToKeepStr: '100', numToKeepStr: '1000']],
          [$class: 'GithubProjectProperty', displayName: '', projectUrlStr: "https://github.awsinternal.tri.global/$githubOrg/${buildEnv.shortName}"],
          disableConcurrentBuilds()
        ]
        slackSend color: 'warning', message: "build $buildLink started", channel: "materials-dev"

        stage('Pre-Checkout') {
        // Check credentials retrieval
        withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'ec2-jenkins-master', variable: 'AWS_ACCESS_KEY_ID']]) {
               sh "echo \"Successfully retrieved AWS credentials\""
        	}
  		   // Checkout Stage:
		// checks out branch that was just updated on GHE
        stage('checkout') {
          checkout scm
          sh "git submodule init"
          sh "git submodule update"
        }

		  // Build Stage:
        // gets aws login and builds docker image using checked out version, tagging as specified in tagCmd

        stage('build') {
				sh "docker build -t $dockerTag -t $dockerRegistry/$dockerRegistryPrefix:$dockerTag -t $dockerRegistry/$dockerRegistryPrefix:$dockerTagLatest -f $dockerfile ."
				echo "Done!"
          slackSend color: 'good', message: "Stage 'build' of build $buildLink passed", channel: "materials-dev"
        }
        stage('test') {
                // Run tests
                withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'ec2-jenkins-master', variable: 'AWS_ACCESS_KEY_ID']]) {
                    sh """
                      docker run --name ${dockerTag}-nosetest \
                          --env AWS_ACCESS_KEY_ID=${env.AWS_ACCESS_KEY_ID} \
                          --env AWS_SECRET_ACCESS_KEY=${env.AWS_SECRET_ACCESS_KEY} \
                          --env AWS_DEFAULT_REGION=us-west-2 \
                          --env CAMD_S3_BUCKET=camd-test \
                          ${dockerTag}
                    """
                    // Retrieve coverage/violations
                    sh """
                      docker cp ${dockerTag}-nosetest:/home/camd/coverage.xml .
                      docker cp ${dockerTag}-nosetest:/home/camd/pylint.out .
                    """
                }
          slackSend color: 'good', message: "Stage 'test' of build $buildLink passed", channel: "materials-dev"
        }

        // Extract analysis from coverage and pylint report
        stage('publish'){
            // TODO: coverage status check on github?
            cobertura coberturaReportFile: 'coverage.xml'
            def pylint_issues = scanForIssues tool: pyLint(pattern: 'pylint.out')
            publishIssues issues: [pylint_issues]
            slackSend color: 'good', message: "Stage 'publish' of build $buildLink passed", channel: "materials-dev"
        }

       	// Push to Docker Stage:
	    // Gets AWS login and pushes docker image to docker registry

        stage('push docker') {

            withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'ec2-jenkins-master', variable: 'AWS_ACCESS_KEY_ID']]) {
                   echo "pushing to $dockerRegistry/$dockerRegistryPrefix:$dockerTag\n\n"
                   sh """
                     export AWS_ACCESS_KEY_ID=${env.AWS_ACCESS_KEY_ID}
                     export AWS_SECRET_ACCESS_KEY=${env.AWS_SECRET_ACCESS_KEY}
                     docker push $dockerRegistry/$dockerRegistryPrefix:$dockerTagLatest
                     docker push $dockerRegistry/$dockerRegistryPrefix:$dockerTag
                     """
                }
            }

          echo "local docker images cleanup\n\n"
          sh """
            # remove old images
            docker images
            docker image prune -a -f --filter "until=240h"  # TODO better filter using labels
            docker container prune --force --filter "until=240h"
            # what's left
            docker images
            """

        slackSend color: 'good', message: "New build ${buildEnv.slackBuildLink} ready for docker pull from ECR: `docker pull $dockerRegistry/$dockerRegistryPrefix:$dockerTag`", channel: "materials-dev"
        }

	  }
	  catch(Exception e) {
        slackSend color: 'danger', message: "Build $buildLink failed", channel: "materials-dev"
        error "error building, ${e.getMessage()}"
      }
    }
  }
}
