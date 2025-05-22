# PowerShell script to rebuild and restart the Docker container

Write-Host "Rebuilding and restarting the Docker container for the deduplication API..."

# Stop any existing container
$containerId = docker ps --filter "publish=8000" --format "{{.ID}}"
if ($containerId) {
    Write-Host "Stopping existing container: $containerId"
    docker stop $containerId
}

# Build the Docker image
Write-Host "Building Docker image..."
docker build -t datacleansing .

# Run the container
Write-Host "Starting new container..."
docker run -d -p 8000:8000 datacleansing

Write-Host "Container rebuilt and restarted successfully."
Write-Host "Wait a few seconds for the API to initialize..."
Start-Sleep -Seconds 5

Write-Host "API should now be available at http://localhost:8000"