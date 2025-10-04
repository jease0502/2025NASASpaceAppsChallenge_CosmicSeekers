export const ApiService = {
    async get(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) { throw new Error(`Server Error: ${response.status}`); }
            return await response.json();
        } catch (error) {
            console.error(`--- ❌ Could not fetch data from ${url}:`, error);
            return null;
        }
    },
    async post(url, body) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const data = await response.json();
            if (!response.ok) {
                console.error("Prediction server returned an error:", data);
                return data;
            }
            return data;
        } catch (error) {
            console.error("--- ❌ Network error or failed to parse JSON:", error);
            return { error: 'Network error or invalid server response.' };
        }
    },
    
    getAllSystems() { return this.get('/get_all_systems'); },
    getTripleSystemData() { return this.get('/get_triple_system_data'); },
    getSpecificTripleSystemData(systemId) { return this.get(`/get_specific_triple_system_data/${systemId}`); },
    getSystemData(systemId) { return this.get(`/get_system_data/${systemId}`); },
    getModelStats() { return this.get('/get_model_stats'); },
    
    predictSystemData(modelName, planets, useLatestTrained = false) { // <<< 修改：增加 useLatestTrained 參數
        const payload = { 
            model_name: modelName, 
            planets: planets,
            use_latest_trained: useLatestTrained // <<< 修改：將參數加入 payload
        };
        return this.post('/predict', payload);
    },

    uploadAndProcess(formData) {
        return fetch('/upload_and_process', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .catch(error => {
            console.error("--- ❌ Network error or failed to parse JSON:", error);
            return { success: false, error: 'Network error or invalid server response.' };
        });
    },

    runTraining(modelChoice, hyperparameters, inputCsvPath) {
        // 這邊的 modelChoice 是 'RandomForest', 'LightGBM' 等
        const payload = {
            model_choice: modelChoice,
            hyperparameters: hyperparameters,
            input_csv_path: inputCsvPath
        };
        return this.post('/run_training', payload);
    },
};