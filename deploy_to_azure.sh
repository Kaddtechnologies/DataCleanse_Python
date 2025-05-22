#!/usr/bin/env pwsh
# Simple deployment script for DuplicateFinder Azure Container App

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroup = "pottersailearning",

    [Parameter(Mandatory=$true)]
    [string]$RegistryName = "kaddacontainerregistry",

    [Parameter(Mandatory=$true)]
    [string]$ContainerAppName = "datacleansing",

    [Parameter(Mandatory=$false)]
    [string]$Location = "centralus",

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest"
)

$ErrorActionPreference = "Stop"
$CurrentDate = Get-Date -Format "yyyyMMdd-HHmmss"
$ImageName = "$ContainerAppName"
$FullImageTag = "$CurrentDate-$ImageTag"

Write-Host "Building and deploying DuplicateFinder application..." -ForegroundColor Cyan

# 1. Check if Azure CLI is installed
try {
    $azVersion = az --version
    Write-Host "Azure CLI is installed" -ForegroundColor Green
}
catch {
    Write-Host "Azure CLI is not installed. Please install it and try again." -ForegroundColor Red
    exit 1
}

# 2. Check Azure login status
try {
    $account = az account show | ConvertFrom-Json
    Write-Host "Logged in to Azure as: $($account.user.name)" -ForegroundColor Green
}
catch {
    Write-Host "Not logged in to Azure. Please login first." -ForegroundColor Red
    Write-Host "Run: az login" -ForegroundColor Yellow
    exit 1
}

# 3. Login to Azure Container Registry
Write-Host "Logging in to Azure Container Registry..." -ForegroundColor Cyan
$acrLoginServer = az acr show --name $RegistryName --query loginServer -o tsv
$acrUsername = az acr credential show --name $RegistryName --query username -o tsv
$acrPassword = az acr credential show --name $RegistryName --query "passwords[0].value" -o tsv

if (-not $acrLoginServer -or -not $acrUsername -or -not $acrPassword) {
    Write-Host "Failed to get ACR credentials" -ForegroundColor Red
    exit 1
}

Write-Host "Successfully retrieved ACR credentials" -ForegroundColor Green

# Login to Docker
Write-Host "Logging in to Docker registry..." -ForegroundColor Cyan
$result = az acr login --name $RegistryName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to login to ACR with the az acr login command" -ForegroundColor Red
    Write-Host "Attempting to login with docker login command..." -ForegroundColor Yellow

    echo $acrPassword | docker login $acrLoginServer --username $acrUsername --password-stdin
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to login to Docker registry" -ForegroundColor Red
        exit 1
    }
}
Write-Host "Successfully logged in to Docker registry" -ForegroundColor Green

# 4. Build Docker image
$fullImageName = "${acrLoginServer}/${ImageName}:${FullImageTag}"
Write-Host "Building Docker image: $fullImageName" -ForegroundColor Cyan
docker build -t $fullImageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to build Docker image" -ForegroundColor Red
    exit 1
}
Write-Host "Successfully built Docker image" -ForegroundColor Green

# 5. Push Docker image to ACR
Write-Host "Pushing Docker image to ACR..." -ForegroundColor Cyan
docker push $fullImageName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to push Docker image to ACR" -ForegroundColor Red
    exit 1
}
Write-Host "Successfully pushed Docker image to ACR" -ForegroundColor Green

# 6. Check if Container App environment exists
$envExists = az containerapp env list --resource-group $ResourceGroup | ConvertFrom-Json
$envName = "duplicate-finder-env"

if ($null -eq $envExists -or $envExists.Count -eq 0) {
    Write-Host "Creating Container App environment..." -ForegroundColor Cyan
    az containerapp env create `
        --name $envName `
        --resource-group $ResourceGroup `
        --location $Location

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create Container App environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Successfully created Container App environment" -ForegroundColor Green
} else {
    $envName = $envExists[0].name
    Write-Host "Using existing Container App environment: $envName" -ForegroundColor Green
}

# 7. Create or update the Container App
$containerAppExists = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Creating new Container App..." -ForegroundColor Cyan
    az containerapp create `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --environment $envName `
        --registry-server $acrLoginServer `
        --registry-username $acrUsername `
        --registry-password $acrPassword `
        --image $fullImageName `
        --target-port 8000 `
        --ingress external `
        --cpu 1.0 `
        --memory 2.0Gi `
        --min-replicas 1 `
        --max-replicas 5 `
        --env-vars "ENVIRONMENT=production" "PORT=8000" "HOST=0.0.0.0" "OPEN-API-KEY=$env:OPEN-API-KEY" `
        --query properties.configuration.ingress.fqdn

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create Container App" -ForegroundColor Red
        exit 1
    }
    Write-Host "Successfully created Container App" -ForegroundColor Green
} else {
    Write-Host "Updating existing Container App..." -ForegroundColor Cyan

    # First update the container app's image
    az containerapp update `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --image $fullImageName

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to update Container App image" -ForegroundColor Red
        exit 1
    }

    # Now update registry credentials separately if needed
    az containerapp registry set `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --server $acrLoginServer `
        --username $acrUsername `
        --password $acrPassword

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to update registry credentials" -ForegroundColor Yellow
    }

    # Set environment variables
    az containerapp update `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --set-env-vars "ENVIRONMENT=production" "PORT=8000" "HOST=0.0.0.0" "OPEN-API-KEY=$env:OPEN-API-KEY"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to update environment variables" -ForegroundColor Yellow
    }

    # Set CPU and memory
    az containerapp update `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --cpu 1.0 `
        --memory 2.0Gi

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to update CPU and memory" -ForegroundColor Yellow
    }

    Write-Host "Successfully updated Container App" -ForegroundColor Green
}

# 8. Get the FQDN
$fqdn = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
Write-Host "Deployment Complete!" -ForegroundColor Green
Write-Host "Application URL: https://$fqdn" -ForegroundColor Cyan
Write-Host "API Documentation: https://$fqdn/docs" -ForegroundColor Cyan
Write-Host "API Root: https://$fqdn/" -ForegroundColor Cyan

# 9. Test API endpoints
Write-Host "Testing API endpoints..." -ForegroundColor Cyan
Write-Host "To test the API, you can use the following commands:" -ForegroundColor Yellow
Write-Host "Upload a file: curl -X POST -F 'file=@sample_data.csv' https://$fqdn/upload" -ForegroundColor Yellow
Write-Host "Get columns: curl https://$fqdn/files/{file_id}/columns" -ForegroundColor Yellow
Write-Host "Set column mapping: curl -X POST -H 'Content-Type: application/json' -d '{\"customer_name\": \"Customer Name\"}' https://$fqdn/files/{file_id}/column-mapping" -ForegroundColor Yellow
Write-Host "Run deduplication: curl -X POST https://$fqdn/files/{file_id}/deduplicate" -ForegroundColor Yellow
Write-Host "Get results: curl https://$fqdn/results/{result_id}" -ForegroundColor Yellow
Write-Host "Export results: curl https://$fqdn/results/{result_id}/export/csv > results.csv" -ForegroundColor Yellow