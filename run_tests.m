function results = run_tests
%RUN_TESTS Run all synthetic dataset and preprocessing tests.
root = fileparts(mfilename('fullpath'));
setup;
results = runtests(fullfile(root, 'tests'));
assert(~isempty(results), 'genderRecognition:MissingTests', 'No tests were discovered.');
fprintf('Executed: %d; passed: %d; failed: %d; incomplete: %d\n', ...
    numel(results), nnz([results.Passed]), nnz([results.Failed]), nnz([results.Incomplete]));
assertSuccess(results);
end
