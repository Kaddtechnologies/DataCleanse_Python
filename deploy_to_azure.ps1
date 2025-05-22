#!/usr/bin/env pwsh
# Simple deployment script for DuplicateFinder Azure Container App

param(
    [Parameter(Mandatory=$false)]
    [string]$ResourceGroup = "pottersailearning",

    [Parameter(Mandatory=$false)]
    [string]$RegistryName = "kaddacontainerregistry",

    [Parameter(Mandatory=$false)]
    [string]$ContainerAppName = "datacleansing",

    [Parameter(Mandatory=$false)]
    [string]$Location = "centralus",

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "latest",
    
    [Parameter(Mandatory=$false)]
    [string]$OpenAIApiKey = [Environment]::GetEnvironmentVariable("OPEN-API-KEY"),
    
    [Parameter(Mandatory=$false)]
    [string]$OpenAIEndpoint = "",
    
    [Parameter(Mandatory=$false)]
    [switch]$UseManagedIdentity = $false,
    
    [Parameter(Mandatory=$false)]
    [string]$OpenAIResourceId = ""
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

# 7. Check if Container App exists and create or update it
$containerAppExists = $false
try {
    $appCheck = az containerapp show --name $ContainerAppName --resource-group $ResourceGroup 2>&1
    if ($LASTEXITCODE -eq 0) {
        $containerAppExists = $true
    }
} catch {
    # App doesn't exist, which is fine
    $containerAppExists = $false
}

# 7.1 Create user-managed identity if using managed identity
$managedIdentityId = ""
if ($UseManagedIdentity) {
    Write-Host "Setting up managed identity for Azure OpenAI access..." -ForegroundColor Cyan
    
    # Check if managed identity already exists
    $identityName = "$ContainerAppName-identity"
    $identityExists = az identity show --name $identityName --resource-group $ResourceGroup 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        # Create managed identity
        Write-Host "Creating managed identity: $identityName" -ForegroundColor Cyan
        $identity = az identity create --name $identityName --resource-group $ResourceGroup | ConvertFrom-Json
        $managedIdentityId = $identity.id
        $principalId = $identity.principalId
        
        # Assign role to the managed identity if OpenAI resource ID is provided
        if ($OpenAIResourceId) {
            Write-Host "Assigning Cognitive Services OpenAI User role to managed identity..." -ForegroundColor Cyan
            az role assignment create --assignee $principalId --scope $OpenAIResourceId --role "Cognitive Services OpenAI User"
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Warning: Failed to assign role to managed identity. Please assign it manually." -ForegroundColor Yellow
            } else {
                Write-Host "Successfully assigned role to managed identity" -ForegroundColor Green
            }
        } else {
            Write-Host "Warning: OpenAI Resource ID not provided. Please assign appropriate role to the managed identity manually." -ForegroundColor Yellow
        }
    } else {
        # Use existing managed identity
        $identity = $identityExists | ConvertFrom-Json
        $managedIdentityId = $identity.id
        Write-Host "Using existing managed identity: $identityName" -ForegroundColor Green
    }
} else {
    # 7.2 Set up the OpenAI API key as a secret if not using managed identity
    Write-Host "Setting up OpenAI API key as a secret..." -ForegroundColor Cyan
    if ($OpenAIApiKey) {
        az containerapp secret set `
            --name $ContainerAppName `
            --resource-group $ResourceGroup `
            --secrets "openai-api-key=$OpenAIApiKey"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Failed to set OpenAI API key as a secret. Will continue with environment variable." -ForegroundColor Yellow
        } else {
            Write-Host "Successfully set OpenAI API key as a secret" -ForegroundColor Green
        }
    } else {
        Write-Host "Warning: OpenAI API key not provided. Make sure it's set in the container app." -ForegroundColor Yellow
    }
}

# Set up environment variables based on authentication method
$envVars = "ENVIRONMENT=production PORT=8000 HOST=0.0.0.0"
if ($UseManagedIdentity) {
    $envVars += " USE_MANAGED_IDENTITY=true"
    if ($OpenAIEndpoint) {
        $envVars += " AZURE_OPENAI_ENDPOINT=$OpenAIEndpoint"
    }
} else {
    $envVars += " OPEN-API-KEY=secretref:openai-api-key"
}

if (-not $containerAppExists) {
    Write-Host "Creating new Container App..." -ForegroundColor Cyan
    
    # Base command for creating container app
    $createCmd = "az containerapp create " +
        "--name $ContainerAppName " +
        "--resource-group $ResourceGroup " +
        "--environment $envName " +
        "--registry-server $acrLoginServer " +
        "--registry-username $acrUsername " +
        "--registry-password $acrPassword " +
        "--image $fullImageName " +
        "--target-port 8000 " +
        "--ingress external " +
        "--cpu 1.0 " +
        "--memory 2.0Gi " +
        "--min-replicas 1 " +
        "--max-replicas 5 "
    
    # Add user-assigned identity if using managed identity
    if ($UseManagedIdentity -and $managedIdentityId) {
        $createCmd += "--user-assigned $managedIdentityId "
    } else {
        $createCmd += "--secrets `"openai-api-key=$OpenAIApiKey`" "
    }
    
    # Add environment variables
    $createCmd += "--env-vars `"$envVars`" " +
        "--query properties.configuration.ingress.fqdn"
    
    # Execute the command
    Invoke-Expression $createCmd
    
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

    # Add user-assigned identity if using managed identity
    if ($UseManagedIdentity -and $managedIdentityId) {
        az containerapp identity assign `
            --name $ContainerAppName `
            --resource-group $ResourceGroup `
            --user-assigned $managedIdentityId
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Failed to assign managed identity to Container App" -ForegroundColor Yellow
        } else {
            Write-Host "Successfully assigned managed identity to Container App" -ForegroundColor Green
        }
    }

    # Set environment variables
    az containerapp update `
        --name $ContainerAppName `
        --resource-group $ResourceGroup `
        --set-env-vars $envVars

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
Write-Host "To test the API, you can use the following command:" -ForegroundColor Yellow
Write-Host "Run deduplication:" -ForegroundColor Yellow
Write-Host "curl -X POST https://$fqdn/deduplicate/ -F `"file=@sample_data.csv`" -F `"column_map_json={`\`"customer_name`\`":`\`"Customer Name`\`",`\`"address`\`":`\`"Address`\`",`\`"city`\`":`\`"City`\`",`\`"country`\`":`\`"Country`\`",`\`"tpi`\`":null}`"" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Yellow
Write-Host "Or for Windows PowerShell:" -ForegroundColor Yellow
Write-Host "Invoke-RestMethod -Uri https://$fqdn/deduplicate/ -Method Post -Form @{file=Get-Item .\sample_data.csv; column_map_json='{`"customer_name`":`"Customer Name`",`"address`":`"Address`",`"city`":`"City`",`"country`":`"Country`",`"tpi`":null}'}" -ForegroundColor Yellow
Write-Host "" -ForegroundColor Yellow
Write-Host "Note: The API uses a single endpoint that accepts both the file and column mapping in one request." -ForegroundColor Yellow
Write-Host "For full API documentation, visit: https://$fqdn/docs" -ForegroundColor Yellow