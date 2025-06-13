// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: 'git@github.com:tingyuliang0116/Multi-Model-Recommendation-System.git' // <-- MODIFIED LINE
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
