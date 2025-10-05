import { ApiService } from '../api.js';
import { AppState } from '../main.js';

const modelConfigs = {
    RandomForest: { n_estimators: { label: 'Number of Trees (n_estimators)', type: 'range', default: 1000, min: 100, max: 3000, step: 50 }, max_depth: { label: 'Max Depth (max_depth)', type: 'range', default: 10, min: 3, max: 50, step: 1 }, min_samples_split: { label: 'Min Samples Split (min_samples_split)', type: 'range', default: 5, min: 2, max: 20, step: 1 }, max_features: { label: 'Max Features (max_features)', type: 'select', options: ['sqrt', 'log2', 'auto'], default: 'sqrt' }, }, GradientBoosting: { n_estimators: { label: 'Number of Trees (n_estimators)', type: 'range', default: 1000, min: 100, max: 3000, step: 50 }, learning_rate: { label: 'Learning Rate (learning_rate)', type: 'range', default: 0.05, min: 0.01, max: 0.3, step: 0.01 }, max_depth: { label: 'Max Depth (max_depth)', type: 'range', default: 5, min: 2, max: 12, step: 1 }, subsample: { label: 'Subsample Ratio (subsample)', type: 'range', default: 0.8, min: 0.5, max: 1.0, step: 0.05 }, }, LightGBM: { n_estimators: { label: 'Number of Trees (n_estimators)', type: 'range', default: 1500, min: 200, max: 4000, step: 100 }, learning_rate: { label: 'Learning Rate (learning_rate)', type: 'range', default: 0.05, min: 0.01, max: 0.3, step: 0.01 }, num_leaves: { label: 'Number of Leaves (num_leaves)', type: 'range', default: 31, min: 10, max: 100, step: 1 }, reg_alpha: { label: 'L1 Regularization (reg_alpha)', type: 'range', default: 0.1, min: 0, max: 1, step: 0.1 }, reg_lambda: { label: 'L2 Regularization (reg_lambda)', type: 'range', default: 0.1, min: 0, max: 1, step: 0.1 }, }, LogisticRegression: { C: { label: 'Inverse Regularization Strength (C)', type: 'range', default: 0.5, min: 0.01, max: 10, step: 0.01 }, penalty: { label: 'Penalty Type (penalty)', type: 'select', options: ['l1', 'l2'], default: 'l1' }, solver: { label: 'Solver (solver)', type: 'select', options: ['saga'], default: 'saga' }, }
};

let localState = {
    eventController: null
};

function renderHyperparameterForm(modelName, container) {
    const config = modelConfigs[modelName];
    let formHTML = '';
    
    for (const param in config) {
        const info = config[param];
        const inputId = `param-${param}`;
        let inputHTML = '';
        
        if (info.type === 'select') {
            const optionsHTML = info.options.map(opt => `<option value="${opt}" ${opt === info.default ? 'selected' : ''}>${opt}</option>`).join('');
            inputHTML = `
                <div class="relative custom-select-wrapper">
                    <select id="${inputId}" data-param="${param}" class="w-full bg-gray-700 border border-gray-600 rounded-md text-sm p-2 appearance-none focus:outline-none focus:ring-2 focus:ring-primary/50">
                        ${optionsHTML}
                    </select>
                    <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-400">
                        <span class="material-symbols-outlined">expand_more</span>
                    </div>
                </div>
            `;
        } else if (info.type === 'range') {
            inputHTML = `
                <div class="slider-container">
                    <input type="range" id="${inputId}" data-param="${param}" value="${info.default}" min="${info.min}" max="${info.max}" step="${info.step}" class="form-slider">
                    <span class="slider-value">${info.default}</span>
                </div>
            `;
        }
        
        formHTML += `<div><label for="${inputId}" class="block text-sm font-medium text-gray-300 mb-1">${info.label}</label>${inputHTML}</div>`;
    }
    
    container.innerHTML = formHTML;

    container.addEventListener('input', (event) => {
        if (event.target.type === 'range') {
            const valueSpan = event.target.nextElementSibling;
            if (valueSpan && valueSpan.classList.contains('slider-value')) {
                valueSpan.textContent = event.target.value;
            }
        }
    }, { signal: localState.eventController.signal });
}

function renderClassificationReport(report) {
    if (!report || typeof report !== 'object') return '';
    const classes = ['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'];
    const rows = classes.map(cls => {
        const m = report[cls] || { precision: 0, recall: 0, ['f1-score']: 0, support: 0 };
        return `<tr>
            <th class="cm-label">${cls}</th>
            <td class="cm-cell">${(m.precision ?? 0).toFixed(3)}</td>
            <td class="cm-cell">${(m.recall ?? 0).toFixed(3)}</td>
            <td class="cm-cell">${(m['f1-score'] ?? 0).toFixed(3)}</td>
            <td class="cm-cell">${m.support ?? 0}</td>
        </tr>`;
    }).join('');

    const macro = report['macro avg'] || {}; const weighted = report['weighted avg'] || {}; const accuracy = report['accuracy'];
    const summary = `
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
            <div class="p-3 rounded-lg bg-[#2E2E2E]/50 border border-white/10">
                <div class="text-sm text-gray-400">Accuracy</div>
                <div class="text-xl font-bold">${(accuracy ?? 0).toFixed(3)}</div>
            </div>
            <div class="p-3 rounded-lg bg-[#2E2E2E]/50 border border-white/10">
                <div class="text-sm text-gray-400">Macro F1</div>
                <div class="text-xl font-bold">${(macro['f1-score'] ?? 0).toFixed(3)}</div>
            </div>
            <div class="p-3 rounded-lg bg-[#2E2E2E]/50 border border-white/10">
                <div class="text-sm text-gray-400">Weighted F1</div>
                <div class="text-xl font-bold">${(weighted['f1-score'] ?? 0).toFixed(3)}</div>
            </div>
        </div>`;

    return `
        <div class="mt-6">
            <h4 class="text-sm font-bold text-gray-400 mb-2">Training Evaluation (Classification Report)</h4>
            ${summary}
            <table class="confusion-matrix-table mt-3">
                <tr>
                    <th class="cm-header cm-header-actual"><span>Class</span></th>
                    <th class="cm-label">Precision</th>
                    <th class="cm-label">Recall</th>
                    <th class="cm-label">F1-Score</th>
                    <th class="cm-label">Support</th>
                </tr>
                ${rows}
            </table>
        </div>
    `;
}

function renderConfusionMatrix(matrixData) {
    if (!matrixData || !Array.isArray(matrixData.matrix)) return '<p>Unable to display confusion matrix.</p>';
    const matrix = matrixData.matrix;
    let labels = Array.isArray(matrixData.labels) ? matrixData.labels : [];
    const n = matrix.length;
    if (!labels || labels.length !== n) {
        labels = Array.from({ length: n }, (_, i) => `Class ${i}`);
    }

    const headerCells = labels.map(label => `<th class="cm-label">${label}</th>`).join('');
    const rowsHtml = matrix.map((row, i) => {
        const cells = row.map(val => `<td class="cm-cell">${val}</td>`).join('');
        return `<tr><th class="cm-label">${labels[i]}</th>${cells}</tr>`;
    }).join('');

    return `
        <div class="mt-4">
            <h4 class="text-sm font-bold text-gray-400 mt-4 mb-2">Training Evaluation (Confusion Matrix)</h4>
            <table class="confusion-matrix-table">
                <tr><td class="cm-empty"></td><th colspan="${n}" class="cm-header">Predicted</th></tr>
                <tr><th class="cm-header cm-header-actual"><span>Actual</span></th>${headerCells}</tr>
                ${rowsHtml}
            </table>
            <p class="text-xs text-gray-500 mt-2">This matrix shows the model performance on the test set. You can now use this trained model for predictions in the Exploration page.</p>
        </div>
    `;
}

export const TrainingSystem = {
    init() {
        let processedFileUrl = null;
        localState.eventController = new AbortController();
        const { signal } = localState.eventController;

        const fileInput = document.getElementById('dataset-file-input');
        const processBtn = document.getElementById('upload-and-process-btn');
        const processResultContainer = document.getElementById('process-result-container');
        
        const trainingStepContainer = document.getElementById('training-step-container');
        const modelSelector = document.getElementById('model-selector');
        const hyperparameterForm = document.getElementById('hyperparameter-form');
        const startTrainingBtn = document.getElementById('start-training-btn');
        const trainingResultContainer = document.getElementById('training-result-container');
        
        // --- ★ START: 新增 JS 邏輯來更新檔名顯示 ---
        const fileNameDisplay = document.getElementById('file-name-display');

        if (fileInput && fileNameDisplay) {
            fileInput.addEventListener('change', () => {
                if (fileInput.files.length > 0) {
                    fileNameDisplay.textContent = fileInput.files[0].name;
                } else {
                    fileNameDisplay.textContent = 'No file selected';
                }
            }, { signal });
        }
        // --- ★ END: 新增 JS 邏輯 ---

        if (processBtn) {
            processBtn.addEventListener('click', async () => {
                const file = fileInput.files[0];
                if (!file) {
                    processResultContainer.innerHTML = `<p class="text-amber-400 text-sm font-bold">Please select a file to upload first.</p>`;
                    return;
                }
                processBtn.disabled = true;
                processBtn.innerHTML = `<span class="material-symbols-outlined animate-spin">autorenew</span> Uploading and processing...`;
                trainingStepContainer.classList.add('hidden');
                processResultContainer.innerHTML = '';
                
                const formData = new FormData();
                formData.append('file', file);
                
                const result = await ApiService.uploadAndProcess(formData);
                
                processBtn.disabled = false;
                processBtn.innerHTML = `<span class="material-symbols-outlined">upload_file</span> Upload and Process`;

                if (result.success) {
                    processedFileUrl = result.file_url;
                    processResultContainer.innerHTML = `<h3 class="text-xl font-bold text-green-400 mb-2">✓ ${result.message || 'Processing successful!'}</h3>`;
                    trainingStepContainer.classList.remove('hidden');
                } else {
                    processResultContainer.innerHTML = `<h3 class="text-xl font-bold text-red-400 mb-2">✗ ${result.error || 'Processing failed'}</h3><p class="text-xs text-gray-400 mt-1">${result.details || ''}</p>`;
                }
            }, { signal });
        }

        if (modelSelector && hyperparameterForm) {
            renderHyperparameterForm(modelSelector.value, hyperparameterForm);
            modelSelector.addEventListener('change', () => renderHyperparameterForm(modelSelector.value, hyperparameterForm), { signal });
        }

        if (startTrainingBtn) {
            startTrainingBtn.addEventListener('click', async () => {
                if (!processedFileUrl) {
                    trainingResultContainer.innerHTML = `<p class="text-red-400">Error: No processed dataset file found.</p>`;
                    return;
                }

                startTrainingBtn.disabled = true;
                startTrainingBtn.innerHTML = `<span class="material-symbols-outlined animate-spin">autorenew</span> Training in progress, please wait...`;
                trainingResultContainer.innerHTML = '';
                
                const modelChoice = modelSelector.value;
                const modelKey = modelChoice.replace('RandomForest', 'random_forest').replace('GradientBoosting', 'gradient_boosting').replace('LightGBM', 'lightgbm').replace('LogisticRegression', 'logistic_regression');

                const hyperparameters = {};
                hyperparameterForm.querySelectorAll('input, select').forEach(input => {
                    hyperparameters[input.dataset.param] = input.value;
                });
                
                const result = await ApiService.runTraining(modelChoice, hyperparameters, processedFileUrl);

                startTrainingBtn.disabled = false;
                startTrainingBtn.innerHTML = `<span class="material-symbols-outlined">model_training</span> Restart Training`;

                if (result.success) {
                    AppState.latestTrainedModelName = modelKey; 
                    console.log(`Set ${modelKey} as the latest temporary model.`);
                    trainingResultContainer.innerHTML = `
                        <h4 class="text-lg font-bold text-green-400 mb-2">✓ ${result.message}</h4>
                        ${renderConfusionMatrix(result.confusion_matrix)}
                        ${renderClassificationReport(result.classification_report || { accuracy: result.accuracy })}
                    `;
                } else {
                    trainingResultContainer.innerHTML = `<h4 class="text-lg font-bold text-red-400 mb-2">✗ Training failed</h4><p class="text-xs text-gray-400">${result.details || result.error}</p>`;
                }
            }, { signal });
        }
    },
    update() {},
    cleanup() {
        console.log("Cleaning up TrainingSystem view...");
        if (localState.eventController) {
            localState.eventController.abort();
            localState.eventController = null;
        }
    }
};