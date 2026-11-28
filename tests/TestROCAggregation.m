classdef TestROCAggregation < matlab.unittest.TestCase
    methods (Test)
        function perfectReversedAndCompleteTies(testCase)
            labels = [0; 0; 1; 1; 1];
            perfect = computeROC(labels, labels);
            reversed = computeROC(labels, -labels);
            ties = computeROC(labels, ones(5, 1));
            testCase.checkCurve(aggregateROCCurves(repmat(perfect, 5, 1)), [0; 0; 1], [0; 1; 1], 1);
            testCase.checkCurve(aggregateROCCurves(repmat(reversed, 5, 1)), [0; 1; 1], [0; 0; 1], 0);
            testCase.checkCurve(aggregateROCCurves(repmat(ties, 5, 1)), [0; 1], [0; 1], 0.5);
            testCase.checkCurve(aggregateROCCurves([perfect; reversed; reversed; perfect; ties]), ...
                [0; 0; 1; 1], [0; 0.4; 0.6; 1], 0.5);
        end

        function exactAreaWithDifferentBreakpointsAndVerticalSegments(testCase)
            % These paths correspond to 4 and 2 negatives, and 5 and 1 positives.
            first = struct('FPR', [0; 0; .25; .25; .75; 1; 1], 'TPR', [0; .2; .4; .6; .8; .8; 1], 'Valid', true);
            second = struct('FPR', [0; .5; .5; 1], 'TPR', [0; 0; 1; 1], 'Valid', true);
            result = aggregateROCCurves([first, second]);
            testCase.checkCurve(result, [0; 0; .25; .25; .5; .5; .75; 1; 1], ...
                [0; .1; .2; .3; .35; .85; .9; .9; 1], (.625 + .5) / 2);
            testCase.verifyEqual(result.Area, mean([trapz(first.FPR, first.TPR), trapz(second.FPR, second.TPR)]), ...
                'AbsTol', 1e-14);
        end

        function invalidFoldsAreVisible(testCase)
            valid = computeROC([0; 1], [0; 1]);
            singleClass = computeROC([1; 1], [0; 1]);
            invalid = valid;
            invalid.TPR(1) = 0.1;
            result = aggregateROCCurves([valid, singleClass, invalid]);
            testCase.verifyEqual(result.NumValidCurves, 1);
            testCase.verifyEqual(result.InvalidCurveIndices, [2; 3]);
            testCase.verifyEqual(result.Area, 1);
            result = aggregateROCCurves(singleClass);
            testCase.verifyEmpty(result.FPR);
            testCase.verifyTrue(isnan(result.Area));
            testCase.verifyEqual(result.NumValidCurves, 0);
        end
    end

    methods (Access = private)
        function checkCurve(testCase, curve, fpr, tpr, area)
            testCase.verifyEqual(curve.FPR, fpr, 'AbsTol', 1e-14);
            testCase.verifyEqual(curve.TPR, tpr, 'AbsTol', 1e-14);
            testCase.verifyEqual(curve.Area, area, 'AbsTol', 1e-14);
            testCase.verifyTrue(issorted(curve.FPR) && issorted(curve.TPR));
            testCase.verifyEqual([curve.FPR(1), curve.TPR(1); curve.FPR(end), curve.TPR(end)], [0 0; 1 1]);
        end
    end
end
