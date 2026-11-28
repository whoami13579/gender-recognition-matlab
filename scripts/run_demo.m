function demo = run_demo(action, options)
%RUN_DEMO Present saved project results and run explicitly requested operations.
%   SETUP; RUN_DEMO opens the menu. RUN_DEMO("quick", Name=Value) is batch safe.
%   See docs/en/demo-and-oral-presentation.md for actions and local paths.
arguments
    action (1,1) string {mustBeMember(action, ["menu", "quick", "results", ...
        "baseline", "train", "models", "image", "independent-status", ...
        "independent", "figures"])} = "menu"
    options.TrainingPath (1,1) string = "train"
    options.TestingPath (1,1) string = "test"
    options.BaselineRun (1,1) string = ""
    options.NestedRun (1,1) string = ""
    options.FinalRun (1,1) string = ""
    options.IndependentAttempt (1,1) string = ""
    options.OutputDirectory (1,1) string = "results/demo"
    options.ImagePath (1,1) string = ""
    options.ImageKind (1,1) string {mustBeMember(options.ImageKind, ["local", "training"])} = "local"
    options.ShowFigures (1,1) logical = false
    options.FoldDetails (1,1) logical = false
    options.FoldROC (1,1) logical = false
    options.Synthetic (1,1) logical = false
end
if action == "menu"
    runMenu(options);
    demo = struct();
else
    demo = execute(action, options);
end
end

function demo = execute(action, options)
demo = struct();
switch action
    case {"quick", "results"}
        demo.Baseline = demoCV("baseline", options, true);
        demo.Nested = demoCV("nested", options, true);
        if action == "quick"
            [demo.Final, pipelines] = demoModels(options, true);
            if strlength(options.ImagePath) > 0
                demo.Prediction = demoImage(pipelines, options);
            end
            fprintf('Independent testing is a separate action (independent-status or independent).\n');
        end
    case "baseline"
        destination = fullfile(finalInternal.resolvePath(options.OutputDirectory), "cross-validation");
        demo.Baseline = run_cross_validation(options.TrainingPath, struct( ...
            'OutputDirectory', destination, 'GeneratePlots', false));
        fprintf('Use BaselineRun to display this saved training-side run.\n');
    case "train"
        baseline = demoCV("baseline", options, false);
        nested = demoCV("nested", options, false);
        destination = fullfile(finalInternal.resolvePath(options.OutputDirectory), "final-evaluation");
        demo.Final = run_final_training(options.TrainingPath, OutputDirectory=destination, ...
            BaselineRun=baseline.OutputDirectory, NestedRun=nested.OutputDirectory, Synthetic=options.Synthetic);
    case "models"
        demo.Final = demoModels(options, true);
    case "image"
        [demo.Final, pipelines] = demoModels(options, false);
        demo.Prediction = demoImage(pipelines, options);
    case {"independent-status", "independent"}
        demo.Final = demoModels(options, false);
        demo.Independent = demoIndependent(demo.Final, options, action == "independent");
        if options.ShowFigures && demo.Independent.Status == "complete"
            ax = plotROCCurves(demo.Independent.Records);
            title(ax, demo.Independent.Context + " ROC");
        end
    case "figures"
        demo.FigureDirectory = demoFigures(options);
end
end

function runMenu(options)
actions = ["quick", "baseline", "train", "models", "image", ...
    "independent-status", "independent", "figures", "results"];
options.ShowFigures = usejava('awt');
while true
    fprintf(['\nGender Recognition Grand Challenge\n' ...
        '1  Quick demonstration with saved results and models\n' ...
        '2  Run fixed baseline five-fold cross-validation\n' ...
        '3  Train final models\n' ...
        '4  Verify and display saved final models\n' ...
        '5  Classify one local image\n' ...
        '6  Display saved independent results or status\n' ...
        '7  Run independent testing if needed\n' ...
        '8  Display and export experimental figures\n' ...
        '9  Inspect saved CV summaries and folds\n' ...
        '0  Exit\n']);
    choice = str2double(input('Choice: ', 's'));
    if choice == 0
        return
    end
    if ~ismember(choice, 1:numel(actions))
        fprintf('Enter a number from 0 to %d.\n', numel(actions));
        continue
    end
    current = options;
    current.FoldDetails = options.FoldDetails || actions(choice) == "results";
    if choice == 5 && strlength(current.ImagePath) == 0
        current.ImagePath = string(input('Local image path (blank to return): ', 's'));
        if strlength(strip(current.ImagePath)) == 0
            continue
        end
    end
    try
        result = execute(actions(choice), current);
        if choice == 2
            options.BaselineRun = result.Baseline.OutputDirectory;
        elseif choice == 3
            options.FinalRun = result.Final.OutputDirectory;
            options.IndependentAttempt = "";
        end
    catch cause
        fprintf(2, '%s\n', cause.message);
    end
end
end
