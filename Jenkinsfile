pipeline {
    agent any
    
    parameters {
        string(name: 'RECIPIENT_EMAIL', defaultValue: 'adityasah712@gmail.com', description: 'Recipient email address')
    }

    environment {
        DOCKER_IMAGE = 'spam-app'
        DOCKER_TAG = 'latest'
        // Removed conflicting RECIPIENT_EMAIL from environment
    }

    stages {
        stage("Welcome message") {
            steps {
                echo "Hello! Pipeline has started for ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}"
                echo "Email notifications will be sent to: ${params.RECIPIENT_EMAIL}"
            }
        }

        stage("Code") {
            steps {
                echo "Cloning code from repository"
                git url: "https://github.com/adityasah104/Spam_detection_API.git", branch: "main"
            }
        }

        stage("Build") {
            steps {
                echo "Building Docker image: ${DOCKER_IMAGE}:${DOCKER_TAG}"
                sh "docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} ."
            }
        }

        stage("Test") {
            steps {
                echo "Running tests..."
                echo "No test cases defined yet"
            }
        }

        stage("Push to DockerHub") {
            steps {
                echo "Pushing image to DockerHub"
                withCredentials([usernamePassword(credentialsId: 'docker-cred',
                    passwordVariable: 'DOCKER_HUB_PASS',
                    usernameVariable: 'DOCKER_HUB_USER')]) {
                    
                    sh """
                        echo "Logging in to DockerHub..."
                        docker login -u ${DOCKER_HUB_USER} -p ${DOCKER_HUB_PASS}
                        
                        echo "Tagging image..."
                        docker tag ${DOCKER_IMAGE}:${DOCKER_TAG} ${DOCKER_HUB_USER}/${DOCKER_IMAGE}:${DOCKER_TAG}
                        
                        echo "Pushing image..."
                        docker push ${DOCKER_HUB_USER}/${DOCKER_IMAGE}:${DOCKER_TAG}
                        
                        echo "Logging out from DockerHub..."
                        docker logout
                    """
                }
            }
        }

        stage("Deploy") {
            steps {
                echo "Deploying application..."
                sh """
                    echo "Stopping existing containers..."
                    docker compose down || true
                    
                    echo "Starting new deployment..."
                    docker compose up -d
                    
                    echo "Checking deployment status..."
                    docker compose ps
                """
            }
        }
    }

    post {
        always {
            echo "Pipeline execution completed"
            echo "Cleaning up workspace..."
            // Optional: Clean up Docker images to save space
            sh 'docker system prune -f || true'
        }

        success {
            emailext (
                subject: "✅ BUILD SUCCESS: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """
                    <html>
                    <body>
                        <h2 style="color: green;">Build Successful! 🎉</h2>
                        
                        <table border="1" cellpadding="5" cellspacing="0">
                            <tr><td><strong>Project:</strong></td><td>${env.JOB_NAME}</td></tr>
                            <tr><td><strong>Build Number:</strong></td><td>${env.BUILD_NUMBER}</td></tr>
                            <tr><td><strong>Duration:</strong></td><td>${currentBuild.durationString}</td></tr>
                            <tr><td><strong>Status:</strong></td><td style="color: green;">SUCCESS</td></tr>
                            <tr><td><strong>Build URL:</strong></td><td><a href="${env.BUILD_URL}">View Build</a></td></tr>
                        </table>
                        
                        <p><strong>Changes:</strong></p>
                        <p>${env.BUILD_URL}changes</p>
                        
                        <p>The application has been successfully deployed!</p>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                to: "${params.RECIPIENT_EMAIL}"
            )
        }

        failure {
            emailext (
                subject: "❌ BUILD FAILED: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """
                    <html>
                    <body>
                        <h2 style="color: red;">Build Failed! ❌</h2>
                        
                        <table border="1" cellpadding="5" cellspacing="0">
                            <tr><td><strong>Project:</strong></td><td>${env.JOB_NAME}</td></tr>
                            <tr><td><strong>Build Number:</strong></td><td>${env.BUILD_NUMBER}</td></tr>
                            <tr><td><strong>Duration:</strong></td><td>${currentBuild.durationString}</td></tr>
                            <tr><td><strong>Status:</strong></td><td style="color: red;">FAILED</td></tr>
                            <tr><td><strong>Build URL:</strong></td><td><a href="${env.BUILD_URL}">View Build</a></td></tr>
                            <tr><td><strong>Console Log:</strong></td><td><a href="${env.BUILD_URL}console">View Console</a></td></tr>
                        </table>
                        
                        <p><strong>Failure Reason:</strong></p>
                        <p>Check the console output for detailed error information.</p>
                        
                        <p>Please investigate and fix the issue.</p>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                to: "${params.RECIPIENT_EMAIL}",
                attachLog: true
            )
        }

        unstable {
            emailext (
                subject: "⚠️ BUILD UNSTABLE: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """
                    <html>
                    <body>
                        <h2 style="color: orange;">Build Unstable! ⚠️</h2>
                        
                        <p>The build completed but some tests failed or there were warnings.</p>
                        
                        <table border="1" cellpadding="5" cellspacing="0">
                            <tr><td><strong>Job:</strong></td><td>${env.JOB_NAME}</td></tr>
                            <tr><td><strong>Build Number:</strong></td><td>${env.BUILD_NUMBER}</td></tr>
                            <tr><td><strong>Status:</strong></td><td style="color: orange;">UNSTABLE</td></tr>
                            <tr><td><strong>Build URL:</strong></td><td><a href="${env.BUILD_URL}">View Build</a></td></tr>
                        </table>
                        
                        <p>Please review the test results and warnings.</p>
                    </body>
                    </html>
                """,
                mimeType: 'text/html',
                to: "${params.RECIPIENT_EMAIL}"
            )
        }
    }
}
