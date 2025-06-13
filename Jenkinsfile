// Jenkinsfile
pipeline {
    agent any // Instructs Jenkins to execute the pipeline on any available agent

    stages {
        stage('Checkout Code') {
            steps {
                git 'git@github.com:tingyuliang0116/Multi-Model-Recommendation-System.git' // Replace with your repo URL
                sh 'ls -l' // List contents to confirm checkout
            }
        }
        stage('Hello World') {
            steps {
                sh 'echo "Hello from Jenkins CI/CD pipeline!"'
            }
        }
    }
}
