model_version=$(jq -r '.model_version | gsub("\\."; "-")' config.json)
port=$(jq -r '.port' config.json)

echo "Building and running house-price-prediction-app-v$model_version"
docker build -t house-price-prediction-app-v$model_version .
docker run -p $port:$port house-price-prediction-app-v$model_version