function results = run_final_evaluation(trainingPath, testingPath, trainingOptions)
%RUN_FINAL_EVALUATION Run final training followed by frozen independent testing.
arguments
    trainingPath (1,1) string = "train"
    testingPath (1,1) string = "test"
    trainingOptions (1,1) struct = struct()
end
options = namedargs2cell(trainingOptions);
training = run_final_training(trainingPath, options{:});
testing = run_independent_testing(training.OutputDirectory, testingPath);
results = struct('Training', training, 'Testing', testing);
end
