# ============================================================================
# Test-OutputQuality.ps1 — Gibberish detection for benchmark results
#
# Checks generated text for signs of MTP collapse: repetitive looping,
# character bursts, low token diversity. Returns $true if output looks
# like real language, $false if it's probably garbage.
#
# Dot-source from benchmark scripts:
#   . "$PSScriptRoot\Test-OutputQuality.ps1"
# ============================================================================

$script:QualityFailDir = $null  # Set before calling to enable logging

function Test-OutputQuality {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [string]$Label = "",  # config label for fail logging (e.g. "n2_p0.3")

        [float]$MinWordRatio = 0.30,
        [float]$MinTrigramRatio = 0.25,
        [int]$MaxBursts = 2
    )

    if ([string]::IsNullOrWhiteSpace($Text)) {
        Write-Warning "[QUALITY] Empty or null output"
        if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_empty.txt" -Encoding utf8 }
        return $false
    }

    $lower = $Text.ToLowerInvariant()
    $words = $lower -split '\s+' | Where-Object { $_ -ne '' }
    $chars = $lower.ToCharArray()
    $wordCount = $words.Count

    if ($wordCount -lt 3) {
        Write-Warning "[QUALITY] Suspiciously short output: $wordCount words"
        return $false
    }

    # --- Check 1: Unique word ratio ---
    $uniqueWords = $words | Select-Object -Unique
    $uniqueCount = $uniqueWords.Count
    $wordRatio = $uniqueCount / [math]::Max($wordCount, 1)
    if ($wordRatio -lt $MinWordRatio) {
        if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_lowwordratio.txt" -Encoding utf8 }
            Write-Warning "[QUALITY] FAIL: Low unique word ratio $([math]::Round($wordRatio,3)) (need > $MinWordRatio)"
        return $false
    }

    # --- Check 2: Word repetition bursts (same word 3+ times consecutively) ---
    $burstCount = 0
    $i = 0
    while ($i -lt $wordCount - 2) {
        $w = $words[$i]
        if ($words[$i+1] -eq $w -and $words[$i+2] -eq $w) {
            $burstCount++
            $i += 3
        } else {
            $i++
        }
    }
    if ($burstCount -gt $MaxBursts) {
        if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_wordburst.txt" -Encoding utf8 }
            Write-Warning "[QUALITY] FAIL: $burstCount word repetition bursts (max $MaxBursts allowed)"
        return $false
    }

    # --- Check 3: Character-level runs (same char 8+ times in a row) ---
    $charCount = $chars.Count
    $foundCharRun = $false
    $i = 0
    while ($i -lt $charCount - 7) {
        $c = $chars[$i]
        $allSame = $true
        for ($j = 1; $j -lt 8; $j++) {
            if ($chars[$i+$j] -ne $c) { $allSame = $false; break }
        }
        if ($allSame) { $foundCharRun = $true; break }
        $i++
    }
    if ($foundCharRun) {
        if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_charrun.txt" -Encoding utf8 }
            Write-Warning "[QUALITY] FAIL: Long character repetition detected (8+ same chars)"
        return $false
    }

    # --- Check 4: Trigram diversity ---
    if ($wordCount -ge 10) {
        $trigrams = @()
        for ($ti = 0; $ti -lt $wordCount - 2; $ti++) {
            $t0 = $words[$ti]
            $t1 = $words[$ti+1]
            $t2 = $words[$ti+2]
            $trigrams += "$t0|$t1|$t2"
        }
        $uniqueTrigrams = $trigrams | Select-Object -Unique
        $utCount = $uniqueTrigrams.Count
        $tCount = $trigrams.Count
        $trigramRatio = $utCount / [math]::Max($tCount, 1)
        if ($trigramRatio -lt $MinTrigramRatio) {
            if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_lowtrigram.txt" -Encoding utf8 }
            Write-Warning "[QUALITY] FAIL: Low unique trigram ratio $([math]::Round($trigramRatio,3)) (need > $MinTrigramRatio)"
            return $false
        }
    }

    # --- Check 5: Punctuation burst detection ---
    $punctChars = @()
    foreach ($c in $chars) {
        if ($c -match '[^a-zA-Z0-9\s]') {
            $punctChars += $c
        }
    }
    $pCount = $punctChars.Count
    if ($pCount -gt 0) {
        $foundPunctRun = $false
        $pi = 0
        while ($pi -lt $pCount - 5) {
            $pc = $punctChars[$pi]
            $allSame = $true
            for ($pj = 1; $pj -lt 6; $pj++) {
                if ($punctChars[$pi+$pj] -ne $pc) { $allSame = $false; break }
            }
            if ($allSame) { $foundPunctRun = $true; break }
            $pi++
        }
        if ($foundPunctRun) {
            if ($script:QualityFailDir -and $Label) { $Text.Substring(0,[math]::Min(500,$Text.Length)) | Out-File -FilePath "$script:QualityFailDir\${Label}_punctburst.txt" -Encoding utf8 }
            Write-Warning "[QUALITY] FAIL: Punctuation repetition burst detected"
            return $false
        }
    }

    return $true
}
