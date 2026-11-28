function records = loadEvaluationResults(path)
%LOADEVALUATIONRESULTS Reconstruct schema-v1 records from the JSON exporter.
records = jsondecode(fileread(path));
if ~isstruct(records) || isempty(records) || ~isvector(records)
    error('genderRecognition:InvalidEvaluationResult', 'Expected an exported nonempty evaluation record array.');
end
records = records(:);
for idx = 1:numel(records)
    row = records(idx);
    if ~isfield(row, 'SchemaVersion') || row.SchemaVersion ~= 1
        error('genderRecognition:InvalidEvaluationResult', 'Unsupported evaluation schema version.');
    end
    row.Method = string(row.Method);
    row.EvaluationStage = string(row.EvaluationStage);
    roc = row.ROC;
    roc.Status = string(roc.Status);
    roc.FPR = roc.FPR(:);
    roc.TPR = roc.TPR(:);
    roc.Thresholds = roc.Thresholds(:);
    if roc.Valid
        if ~isequal(string(roc.InitialThreshold), "positive-infinity")
            error('genderRecognition:InvalidEvaluationResult', 'Missing positive-infinity ROC sentinel.');
        end
        roc.Thresholds = [Inf; roc.Thresholds];
        area = trapz(roc.FPR, roc.TPR);
        verifyRoundedValue(row.AUC, area);
        row.AUC = area;
    else
        if ~isempty(row.AUC) || ~isempty(roc.InitialThreshold)
            error('genderRecognition:InvalidEvaluationResult', 'Undefined AUC must be encoded as null.');
        end
        row.AUC = NaN;
    end
    row.ROC = rmfield(roc, 'InitialThreshold');
    row.Timing.Units = string(row.Timing.Units);
    for group = ["Training", "Inference", "Stages"]
        for scope = string(fieldnames(row.Timing.(group))).'
            timing = row.Timing.(group).(scope);
            if ~isempty(timing)
                roundTrip = summarizeTimingSamples(timing.SamplesSeconds(:), timing.WarmupRuns, timing.BatchSize);
                for field = setdiff(string(fieldnames(roundTrip)).', ["SamplesSeconds", "BatchSize"])
                    if isempty(roundTrip.(field))
                        if ~isempty(timing.(field))
                            error('genderRecognition:InvalidEvaluationResult', 'Unexpected per-image timing value.');
                        end
                    else
                        verifyRoundedValue(timing.(field), roundTrip.(field));
                    end
                end
                row.Timing.(group).(scope) = roundTrip;
            end
        end
    end
    row.Configuration = restoreOptions(row.Configuration);
    accuracy = trace(row.ConfusionMatrix) / row.NumEvaluationSamples;
    verifyRoundedValue(row.Accuracy, accuracy);
    row.Accuracy = accuracy;
    reduction = 1 - row.NumFeatures / row.NumOriginalFeatures;
    verifyRoundedValue(row.DimensionalityReductionFraction, reduction);
    row.DimensionalityReductionFraction = reduction;
    validateEvaluationResult(row);
    records(idx) = row;
end
end

function options = restoreOptions(options)
for group = ["Preprocessing", "Projection", "Classifier"]
    block = options.(group);
    if ~isempty(block)
        for field = string(fieldnames(block)).'
            value = block.(field);
            if ischar(value)
                block.(field) = string(value);
            elseif ~isempty(value) && any(field == ["ImageSize", "HiddenLayerSizes"])
                block.(field) = reshape(value, 1, []);
            end
        end
        options.(group) = block;
    end
end
end

function verifyRoundedValue(value, expected)
% JSON parsers can differ by a few ULPs. Restore derived values from primitives.
if ~isscalar(value) || ~isfinite(value) || abs(value - expected) > 32 * eps(max(1, abs(expected)))
    error('genderRecognition:InvalidEvaluationResult', 'An exported summary disagrees with its source values.');
end
end
