// define a tag according to the Docker tag rules https://docs.docker.com/engine/reference/commandline/tag/
// the hash sign (#) is problematic when using it in bash, instead of working around this problem, just replace all
// punctuation with dash (-)
def dockerTagWithoutBuildNumber = "public-${env.BRANCH_NAME}".toLowerCase().replaceAll("\\p{Punct}", "-").replaceAll("\\p{Space}", "-")
def dockerTag = "${dockerTagWithoutBuildNumber}-${env.BUILD_NUMBER}"
def dockerTagLatest = "${dockerTagWithoutBuildNumber}-latest"
def awsRegion = "us-west-2"
def githubOrg = "materials"
def dockerRegistry = "251589461219.dkr.ecr.${awsRegion}.amazonaws.com"
def dockerRegistryPrefix = "camd-worker"
def dockerfile = "Dockerfile"
def buildLink = "<${env.BUILD_URL}|${env.JOB_NAME} ${env.BUILD_NUMBER}>"
// Define the python test, coverage, lint procedures
def testProcedure = "nosetests --with-xunit --all-modules --traverse-namespace --with-coverage --cover-package=camd --cover-inclusive"
def coverageProcedure = "python -m coverage xml --include=camd*"
def lintProcedure = "pylint -f parseable -d I0011,R0801 camd | tee pylint.out"

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
        // Ask Westin/AMDD if this section needs to be changed
        withCredentials([string(credentialsId: 'vault_token', variable: 'VAULT_TOKEN'),
                         string(credentialsId: 'vault_ip', variable: 'VAULT_IP'),
                         string(credentialsId: 'jumpbox_ip', variable: 'JUMPBOX')]) {

                def VAULT_PORT = "8200"

                echo 'Testing..'
                sh 'whoami'
                sh """
                  eval \"\$(ssh-agent -s)\"
                  ssh-add /var/lib/jenkins/vault.pem
                  ssh -M -S vault-socket -fnNT -A -L $VAULT_PORT:$VAULT_IP:$VAULT_PORT ubuntu@$JUMPBOX
                """
                sh """
                  ssh -S vault-socket -O check ubuntu@$JUMPBOX
                  curl --header "X-Vault-Token: $VAULT_TOKEN"  http://127.0.0.1:8200/v1/aws/creds/beep-ci > temp
                  sleep 10s
                  ssh -S vault-socket -O exit ubuntu@$JUMPBOX
                  python3 ~/json-cred.py
                """
            }
        }

  		  // Checkout Stage:
		  // checks out branch that was just updated on GHE
        stage('checkout') {
          checkout scm
        }

		  // Build Stage:
        // gets aws login and builds docker image using checked out version, tagging as specified in tagCmd

        stage('build') {
				sh "sudo docker build -t $dockerTag -t $dockerRegistry/$dockerRegistryPrefix:$dockerTag -t $dockerRegistry/$dockerRegistryPrefix:$dockerTagLatest -f $dockerfile ."
				echo "Done!"
          slackSend color: 'good', message: "Stage 'build' of build $buildLink passed", channel: "materials-dev"
        }
        stage('test') {
				// Run the tests in a docker container
				// TODO We should write credentials directly into docker
				// ie. write: openssl rand -base64 32 | docker secret create secure-secret -
				// read: docker service create --secret="secure-secret" redis:alpine
				// # cat /run/secrets/secure-secret
                //  echo eval \"\$(aws configure get aws_access_key_id)\"
                //  AWS_ACCESS_KEY_ID=\$(aws configure get aws_access_key_id)
                //  AWS_SECRET_ACCESS_KEY=\$(aws configure get aws_secret_access_key)
                //--env AWS_ACCESS_KEY_ID=\$AWS_ACCESS_KEY_ID --env AWS_SECRET_ACCESS_KEY=\$AWS_SECRET_ACCESS_KEY
                // Run tests
                sh """
                  sudo docker run --name ${dockerTag}-nosetests ${dockerTag} \
                      /bin/bash -c "$testProcedure && $coverageProcedure && $lintProcedure"
                """
                // Retrieve coverage/violations
                sh """
                  sudo docker cp ${dockerTag}-nosetests:/home/camd/coverage.xml .
                  sudo docker cp ${dockerTag}-nosetests:/home/camd/pylint.out .
                """
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
	    // TODO: fix docker push

        // stage('push docker') {

        //   echo "Logging in to ECR in region $awsRegion"
        //   sh """
        //     LOCAL=`aws ecr get-login --region $awsRegion --no-include-email`
        //     sudo \$LOCAL
        //   """

        //   echo "pushing to $dockerRegistry/$dockerRegistryPrefix:$dockerTag\n\n"
        //   sh """
        //     sudo docker push $dockerRegistry/$dockerRegistryPrefix:$dockerTagLatest
        //     sudo docker push $dockerRegistry/$dockerRegistryPrefix:$dockerTag
        //     """

        //   echo "local docker images cleanup\n\n"
        //   sh """
        //     # remove old images
        //     sudo docker images
        //     sudo docker image prune -a -f --filter "until=240h"  # TODO better filter using labels
        //     # what's left
        //     sudo docker images
        //     """

        // slackSend color: 'good', message: "New build ${buildEnv.slackBuildLink} ready for docker pull from ECR: `docker pull $dockerRegistry/$dockerRegistryPrefix:$dockerTag`", channel: "materials-dev"
        // }

	  }
	  catch(Exception e) {
        slackSend color: 'danger', message: "Build $buildLink failed", channel: "materials-dev"
        error "error building, ${e.getMessage()}"
      }
    }
  }
}
