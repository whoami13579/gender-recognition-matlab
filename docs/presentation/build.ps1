param(
    [string] $OutputDirectory = 'results/presentation',
    [switch] $Render
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path (Split-Path $PSScriptRoot)
if (![IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $repo $OutputDirectory
}
$out = (New-Item -ItemType Directory -Path $OutputDirectory -Force).FullName
$slides = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'slides.json') | ConvertFrom-Json
$evidence = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot 'evidence.json') | ConvertFrom-Json
$notesPath = Join-Path $repo 'docs/zh-TW/oral-presentation-notes.md'
$notes = (Get-Content -Raw -Encoding UTF8 $notesPath) -split '(?m)^## (?=\d+\.)'

function Color([string]$hex) {
    return [Convert]::ToInt32($hex.Substring(0,2),16) +
        256*[Convert]::ToInt32($hex.Substring(2,2),16) +
        65536*[Convert]::ToInt32($hex.Substring(4,2),16)
}

$navy = Color '142843'
$ink = Color '24384F'
$muted = Color '486078'
$teal = Color '007C74'
$paper = Color 'F7F9FC'
$white = Color 'FFFFFF'
$line = Color 'CDD7E2'
$pale = Color 'E7F2F1'
$methods = @((Color '0666A9'), (Color 'A34E18'), (Color '11726A'), (Color '7756A1'))

function Format-Text($shape, $value, $size, $color, $bold = $false, $alignment = 1) {
    $frame = $shape.TextFrame
    $range = $frame.TextRange
    $range.Text = $value
    $range.Font.Name = 'Aptos'
    $range.Font.NameFarEast = 'Microsoft JhengHei'
    $range.Font.Size = [single]$size
    $range.Font.Color.RGB = $color
    if ($bold) { $range.Font.Bold = -1 }
    $range.ParagraphFormat.Alignment = $alignment
}

function Add-Text($slide, $value, $x, $y, $w, $h, $size, $color, $bold = $false, $alignment = 1) {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    Format-Text $shape $value $size $color $bold $alignment
}

function Add-Rule($slide, $x1, $y1, $x2, $y2, $color, $weight = 1, $dash = 1, [switch]$Arrow) {
    $shape = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $shape.Line.ForeColor.RGB = $color
    $shape.Line.Weight = [single]$weight
    $shape.Line.DashStyle = $dash
    if ($Arrow) { $shape.Line.EndArrowheadStyle = 3 }
}

function Add-Node($slide, $value, $x, $y, $w, $h, $fill = $pale, $foreground = $navy, $size = 20) {
    $shape = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    $shape.Fill.Solid()
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.Visible = 0
    $shape.TextFrame.VerticalAnchor = 3
    Format-Text $shape $value $size $foreground $true 2
}

function Add-Body($slide, $lines, $y = 146, $step = 80, $size = 24, $width = 835) {
    foreach ($entry in $lines) {
        Add-Text $slide '•' 52 $y 20 30 $size $teal $true
        Add-Text $slide $entry 80 $y $width ($step - 5) $size $ink
        $y += $step
    }
}

function Add-Table($slide, $headers, $rows, $widths, $y = 142, $rowHeight = 48, $size = 18) {
    $count = $rows.Count + 1
    $shape = $slide.Shapes.AddTable($count, $headers.Count, 46, $y, 868, ($count * $rowHeight))
    $table = $shape.Table
    for ($col = 1; $col -le $headers.Count; $col++) {
        $table.Columns.Item($col).Width = [single]$widths[$col - 1]
    }
    for ($row = 1; $row -le $count; $row++) {
        $table.Rows.Item($row).Height = [single]$rowHeight
        for ($col = 1; $col -le $headers.Count; $col++) {
            $cell = $table.Cell($row, $col).Shape
            $cell.Fill.Solid()
            $cell.TextFrame.VerticalAnchor = 3
            $cell.TextFrame.MarginLeft = 10
            $cell.TextFrame.MarginRight = 7
            if ($row -eq 1) {
                $cell.Fill.ForeColor.RGB = $navy
                Format-Text $cell $headers[$col-1] $size $white $true
            } else {
                $fill = if ($row % 2 -eq 0) { $white } else { $pale }
                $cell.Fill.ForeColor.RGB = $fill
                Format-Text $cell ([string]$rows[$row-2][$col-1]) $size $ink ($col -eq 1)
            }
        }
    }
}

function Add-Diagram($slide, $diagram) {
    switch ($diagram) {
        'pipeline' {
            Add-Node $slide '原始影像' 46 247 112 55
            Add-Node $slide "共用前處理`n4,096 維" 188 239 152 71
            Add-Rule $slide 158 274 188 274 $muted 2 -Arrow
            Add-Node $slide 'PCA' 390 162 143 55
            Add-Node $slide 'Fisher LDA' 390 326 143 55
            Add-Rule $slide 340 259 390 190 $muted 2 -Arrow
            Add-Rule $slide 340 290 390 353 $muted 2 -Arrow
            $labels = @('PCA + Bayes', 'PCA + MLP', 'LDA + Bayes', 'LDA + MLP')
            $palette = @($methods[0], $methods[2], $methods[1], $methods[3])
            for ($i = 0; $i -lt 4; $i++) {
                $y = 126 + 83 * $i
                $from = if ($i -lt 2) { 190 } else { 353 }
                Add-Rule $slide 533 $from 619 ($y + 26) $muted 2 -Arrow
                Add-Node $slide $labels[$i] 619 $y 250 52 $palette[$i] $white
            }
        }
        'pixels' {
            Add-Text $slide '前處理流程示意' 46 155 868 28 17 $muted
            Add-Node $slide "RGB 輸入`n640 × 640 × 3" 46 216 253 113
            Add-Rule $slide 299 272 350 272 $muted 2 -Arrow
            Add-Node $slide "灰階轉換與縮放`n64 × 64" 350 216 253 113
            Add-Rule $slide 603 272 654 272 $muted 2 -Arrow
            Add-Node $slide "依 MATLAB 欄序攤平`n4,096 維" 654 216 260 113 $navy $white
        }
        'mlp' {
            Add-Text $slide '凍結架構' 46 145 868 28 17 $muted
            Add-Node $slide "標準化特徵`nPCA 20 維／LDA 1 維" 46 194 245 86
            Add-Rule $slide 291 237 354 237 $muted 2 -Arrow
            Add-Node $slide "tanh 隱藏層`nPCA 16／LDA 8 神經元" 354 194 249 86
            Add-Rule $slide 603 237 666 237 $muted 2 -Arrow
            Add-Node $slide "線性輸出`n1 個 logit" 666 194 248 86 $navy $white
        }
        'folds' {
            Add-Text $slide '示意：其中一折，外層驗證位置依序輪替' 46 150 868 30 18 $muted
            for ($i = 0; $i -lt 5; $i++) {
                $x = 46 + 175*$i
                if ($i -eq 4) { Add-Node $slide "驗證`n40 張" $x 194 168 88 $navy $white }
                else { Add-Node $slide "擬合`n40 張" $x 194 168 88 }
            }
        }
        'nested' {
            Add-Text $slide '外層五折中的一折' 46 146 500 30 18 $muted
            Add-Node $slide "外層訓練區 · 160 張`n內層每折 120 張擬合、40 張選參驗證" 46 192 542 93
            Add-Rule $slide 588 239 640 239 $muted 2 -Arrow
            Add-Node $slide "外層驗證區 · 40 張`n評估重新擬合的勝出模型" 640 192 274 93 $navy $white 18
        }
    }
}

function Add-RocPanel($slide, $caption, $x, $mode) {
    $plotX = $x + 48
    $plotY = 193
    $extent = 192
    Add-Text $slide $caption $x 127 395 32 20 $navy $true 2
    Add-Rule $slide $plotX $plotY $plotX ($plotY+$extent) $muted
    Add-Rule $slide $plotX ($plotY+$extent) ($plotX+$extent) ($plotY+$extent) $muted
    Add-Rule $slide $plotX ($plotY+$extent) ($plotX+$extent) $plotY $muted 1.5 4
    # The eight saved mean curves share the same geometry.
    $dashStyles = @(1, 4, 5, 3)
    $weights = @(5, 4, 3, 1.8)
    $points = $evidence.rocVertices
    for ($method = 0; $method -lt $methods.Count; $method++) {
        for ($v = 1; $v -lt $points.Count; $v++) {
            $a = $points[$v-1]
            $b = $points[$v]
            $x1 = $plotX+$extent*$a[0]
            $x2 = $plotX+$extent*$b[0]
            $y1 = $plotY+$extent*(1-$a[1])
            $y2 = $plotY+$extent*(1-$b[1])
            Add-Rule $slide $x1 $y1 $x2 $y2 $methods[$method] $weights[$method] $dashStyles[$method]
        }
    }
    Add-Text $slide '1' ($plotX-24) ($plotY-11) 19 23 14 $muted
    Add-Text $slide '0' ($plotX-9) ($plotY+$extent+6) 20 22 14 $muted
    Add-Text $slide '1' ($plotX+$extent-5) ($plotY+$extent+6) 20 22 14 $muted
    Add-Text $slide 'TPR' ($plotX-53) 266 45 24 16 $muted
    Add-Text $slide 'FPR' ($plotX+63) 400 60 24 16 $muted $false 2
    for ($method = 0; $method -lt $methods.Count; $method++) {
        $labelY = 226 + 34*$method
        Add-Rule $slide ($plotX+205) $labelY ($plotX+230) $labelY $methods[$method] 2 $dashStyles[$method]
        $row = $evidence.methods[$method]
        $label = '{0} · {1:F3}' -f $row.name, $row.$mode[2]
        Add-Text $slide $label ($plotX+235) ($labelY-11) 139 24 14 $ink
    }
}

function Add-ProjectionSketch($slide, $mode) {
    Add-Text $slide '投影概念示意' 638 206 276 25 16 $muted
    Add-Rule $slide 646 407 877 407 $line
    Add-Rule $slide 646 407 646 247 $line
    $points = @(
        @(673,347), @(690,327), @(711,358), @(732,332), @(751,350), @(774,329),
        @(712,305), @(734,285), @(757,304), @(779,277), @(801,296), @(824,272)
    )
    for ($i = 0; $i -lt $points.Count; $i++) {
        $dot = $slide.Shapes.AddShape(9, $points[$i][0], $points[$i][1], 10, 10)
        $dot.Fill.Solid()
        $dot.Fill.ForeColor.RGB = if ($mode -eq 'pca') { $muted }
            elseif ($i -lt 6) { $methods[0] } else { $methods[1] }
        $dot.Line.Visible = 0
    }
    if ($mode -eq 'pca') {
        Add-Rule $slide 663 382 859 263 $teal 2 -Arrow
        Add-Text $slide '最大變異方向' 753 373 145 26 15 $teal $true
    } else {
        Add-Rule $slide 835 372 779 237 $teal 2 -Arrow
        Add-Text $slide '類別分離方向' 662 375 142 26 15 $teal $true
    }
}

function Add-Evidence($slide, $kind) {
    $rows = @()
    foreach ($m in $evidence.methods) {
        switch ($kind) {
            'complexity' {
                $rows += ,@($m.name, ('{0} / {1:N0}' -f $m.baseline[4],$m.baseline[5]),
                    ('{0} / {1:N0}' -f $m.nested[4],$m.nested[5]), ('{0} / {1:N0}' -f $m.final[0],$m.final[3]))
            }
            'timings' {
                $rows += ,@($m.name, ('{0:F3} / {1:F2}' -f $m.baseline[6],$m.baseline[7]),
                    ('{0:F3} / {1:F2}' -f $m.baseline[8],$m.baseline[9]),
                    ('{0:F3} / {1:F2}' -f $m.nested[6],$m.nested[7]),
                    ('{0:F3} / {1:F2}' -f $m.nested[8],$m.nested[9]),
                    ('{0:F3} / {1:F2}' -f $m.final[5],$m.final[6]))
            }
            'final' {
                $rows += ,@($m.name, $m.setting, ('{0:N0}' -f $m.final[1]), $m.final[2], ('{0:N0}' -f $m.final[4]))
            }
        }
    }
    switch ($kind) {
        'complexity' {
            Add-Table $slide @('方法','Baseline','Nested 外層','最終模型') $rows @(220,216,216,216) 151 46 20
            Add-Text $slide '完整參數量包含投影係數、MLP 標準化係數與分類器。投影包含訓練平均。' 46 402 868 43 20 $ink
        }
        'timings' {
            Add-Table $slide @('方法','Baseline 擬合','Baseline 推論','Nested 擬合','Nested 推論','最終擬合') $rows @(132,147,147,147,147,148) 148 43 16
            $caption = 'CV 擬合每折 160 張，推論每批 {0} 張。最終以 200 張擬合。計時不含磁碟讀取。' -f $evidence.baselineBatch
            Add-Text $slide $caption 46 380 868 52 19 $ink
        }
        'final' {
            Add-Table $slide @('方法','凍結設定','投影','正規化','合計') $rows @(167,324,121,128,128) 149 44 17
            Add-Text $slide ('最終訓練版本 {0}。載入時驗證模型及中繼資料校驗碼。' -f $evidence.finalRevision) 46 389 870 48 20 $ink
        }
    }
}

$app = New-Object -ComObject PowerPoint.Application
$ppt = $app.Presentations.Add(0)
try {
    $ppt.PageSetup.SlideWidth = 960
    $ppt.PageSetup.SlideHeight = 540
    for ($idx = 0; $idx -lt $slides.Count; $idx++) {
        $spec = $slides[$idx]
        $slide = $ppt.Slides.Add($idx+1, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.Solid()
        $slide.Background.Fill.ForeColor.RGB = $paper
        if ($spec.kind -eq 'cover') {
            $slide.Background.Fill.ForeColor.RGB = $navy
            Add-Text $slide $spec.section 65 86 830 35 24 (Color 'A7DAD5') $true
            Add-Text $slide $spec.title 62 171 835 90 67 $white $true
            Add-Text $slide $spec.subtitle 66 273 820 49 34 $white
            Add-Rule $slide 66 365 894 365 (Color '67A5A7') 2
            Add-Text $slide ($spec.body -join '   ·   ') 66 404 825 48 24 $white
        } else {
            Add-Text $slide $spec.section 46 36 868 27 17 $teal $true
            Add-Text $slide $spec.title 46 72 868 57 36 $navy $true
            Add-Rule $slide 46 503 914 503 $line
            Add-Text $slide '圖形識別期末專題' 46 512 400 21 12 $muted
            Add-Text $slide ('{0:00} / {1}' -f ($idx+1), $slides.Count) 822 512 92 21 12 $muted $false 3
            if ($spec.key) { Add-Text $slide $spec.key 46 454 868 45 19 $teal $true }
            switch ($spec.kind) {
                'table' {
                    $widths = if ($spec.headers.Count -eq 4) { @(235,225,205,203) }
                        elseif ($spec.rows.Count -gt 2) { @(159,352,357) } else { @(274,297,297) }
                    Add-Table $slide $spec.headers $spec.rows $widths 149 54 20
                    if ($spec.body) { Add-Body $slide $spec.body 331 62 22 }
                }
                'formula' {
                    Add-Text $slide $spec.formula 46 141 868 57 32 $teal $true
                    if ($spec.diagram) {
                        Add-Body $slide $spec.body 221 72 21 523
                        Add-ProjectionSketch $slide $spec.diagram
                    } else {
                        Add-Body $slide $spec.body 219 71 22
                    }
                }
                'diagram' {
                    Add-Diagram $slide $spec.diagram
                    if ($spec.body) { Add-Body $slide $spec.body 320 68 22 }
                }
                'roc' {
                    Add-RocPanel $slide 'Baseline 五折' 58 baseline
                    Add-RocPanel $slide 'Nested 外層五折' 491 nested
                    Add-Text $slide ('灰虛線：隨機排序。圖例為平均 AUC。折間 SD 均為 0。訓練端版本 {0}。' -f $evidence.cvRevision) 65 432 858 23 15 $muted
                }
                { $_ -in @('complexity','timings','final') } { Add-Evidence $slide $_ }
                'demo' {
                    $commands = @('setup;', 'a = namedargs2cell(demoOptions);',
                        'run_demo("quick", a{:}, ShowFigures=true);',
                        'run_demo("figures", a{:}, ShowFigures=true);', 'run_demo("independent-status", a{:});')
                    $code = $slide.Shapes.AddTextbox(1, 46, 158, 868, 222)
                    $code.Fill.Solid()
                    $code.Fill.ForeColor.RGB = $navy
                    $code.TextFrame.MarginLeft = 20
                    $code.TextFrame.MarginTop = 14
                    Format-Text $code ($commands -join "`r") 25 $white
                    $code.TextFrame.TextRange.Font.Name = 'Consolas'
                    Add-Text $slide 'demoOptions 依操作指南填入已選定的實驗目錄及一張本機影像。' 46 397 868 42 20 $ink
                }
                'references' {
                    Add-Body $slide $spec.body 145 72 21
                    $links = @(
                        @('MathWorks svd', 'https://www.mathworks.com/help/matlab/ref/double.svd.html'),
                        @('MathWorks chol', 'https://www.mathworks.com/help/matlab/ref/chol.html'),
                        @('MathWorks imresize', 'https://www.mathworks.com/help/matlab/ref/imresize.html'),
                        @('專案講義索引', 'https://github.com/whoami13579/gender-recognition-matlab/blob/main/docs/zh-TW/lecture-reference.md')
                    )
                    for ($i = 0; $i -lt $links.Count; $i++) {
                        Add-Text $slide $links[$i][0] (46+211*$i) 449 205 35 16 $teal $true
                        $shape = $slide.Shapes.Item($slide.Shapes.Count)
                        $shape.TextFrame.TextRange.ActionSettings.Item(1).Hyperlink.Address = $links[$i][1]
                    }
                }
                default { Add-Body $slide $spec.body }
            }
        }
        if ($notes.Count -gt ($idx + 1)) {
            $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $notes[$idx+1].Trim()
        }
    }
    $stem = Join-Path $out 'gender-recognition-presentation'
    $ppt.SaveAs($stem + '.pptx', 24)
    $ppt.SaveCopyAs($stem + '.pdf', 32)
    Copy-Item -LiteralPath $notesPath -Destination (Join-Path $out 'speaker-notes.md') -Force
    if ($Render) {
        $folder = (New-Item -ItemType Directory -Path (Join-Path $out 'render') -Force).FullName
        $ppt.Export($folder, 'PNG', 1280, 720)
    }
    Write-Output ('Built {0} editable slides in {1}' -f $slides.Count, $out)
} finally {
    $ppt.Close()
    if ($app.Presentations.Count -eq 0) { $app.Quit() }
}
