#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const root = path.resolve(__dirname, '..');
const coverageDir = path.join(root, '.v8coverage');
const libDir = path.join(root, 'lib');

function walkJsFiles(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...walkJsFiles(full));
      continue;
    }
    if (entry.isFile() && full.endsWith('.js')) out.push(full);
  }
  return out;
}

function runTestsWithCoverage() {
  fs.rmSync(coverageDir, { recursive: true, force: true });
  fs.mkdirSync(coverageDir, { recursive: true });

  const jasmineBin = path.join(root, 'node_modules', 'jasmine', 'bin', 'jasmine.js');
  if (!fs.existsSync(jasmineBin)) {
    throw new Error('Cannot find jasmine binary at ' + jasmineBin);
  }

  const res = cp.spawnSync(
    process.execPath,
    [jasmineBin, '--config=test/jasmine.json'],
    {
      cwd: root,
      stdio: 'inherit',
      env: { ...process.env, NODE_V8_COVERAGE: coverageDir }
    }
  );

  if (res.status !== 0) {
    process.exit(res.status || 1);
  }
}

function coverageFilePath(url) {
  if (url.startsWith('file://')) {
    return decodeURIComponent(new URL(url).pathname);
  }
  return url;
}

function collectRangesByFile() {
  const merged = new Map();
  const coverageFiles = fs.readdirSync(coverageDir).filter((name) => name.endsWith('.json'));

  for (const file of coverageFiles) {
    const payload = JSON.parse(fs.readFileSync(path.join(coverageDir, file), 'utf8'));
    for (const entry of payload.result || []) {
      const filePath = coverageFilePath(entry.url || '');
      if (!filePath.startsWith(libDir + path.sep)) continue;
      if (!filePath.endsWith('.js')) continue;

      let rangeMap = merged.get(filePath);
      if (!rangeMap) {
        rangeMap = new Map();
        merged.set(filePath, rangeMap);
      }

      for (const fn of entry.functions || []) {
        for (const range of fn.ranges || []) {
          const key = range.startOffset + ':' + range.endOffset;
          const prev = rangeMap.get(key) || 0;
          if (range.count > prev) rangeMap.set(key, range.count);
        }
      }
    }
  }

  return merged;
}

function isCodeLine(line) {
  const t = line.trim();
  if (!t) return false;
  if (t.startsWith('//')) return false;
  if (t.startsWith('/*')) return false;
  if (t.startsWith('*')) return false;
  if (t.startsWith('*/')) return false;
  return true;
}

function innermostRangeCount(ranges, offset) {
  let best = null;
  for (const range of ranges) {
    if (range.start <= offset && offset < range.end) {
      const span = range.end - range.start;
      if (!best || span < best.span || (span === best.span && range.start >= best.start)) {
        best = { ...range, span };
      }
    }
  }
  return best ? best.count : null;
}

function measureLineCoverage(filePath, rangeMap) {
  const source = fs.readFileSync(filePath, 'utf8');
  const lines = source.split('\n');
  const ranges = [];

  if (rangeMap) {
    for (const [key, count] of rangeMap.entries()) {
      const [start, end] = key.split(':').map((v) => parseInt(v, 10));
      ranges.push({ start, end, count });
    }
  }

  let covered = 0;
  let executable = 0;
  const uncoveredLines = [];
  let runningOffset = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const firstNonWs = line.search(/\S/);

    if (isCodeLine(line) && firstNonWs >= 0) {
      const probe = runningOffset + firstNonWs;
      const count = innermostRangeCount(ranges, probe);
      executable += 1;
      if (count && count > 0) {
        covered += 1;
      } else {
        uncoveredLines.push(i + 1);
      }
    }

    runningOffset += line.length + 1;
  }

  return { covered, executable, uncoveredLines };
}

function reportAndValidate() {
  const libFiles = walkJsFiles(libDir).sort();
  const mergedRanges = collectRangesByFile();

  let totalCovered = 0;
  let totalExecutable = 0;
  let hasGap = false;

  console.log('\nCoverage (lib/**/*.js):');

  for (const filePath of libFiles) {
    const result = measureLineCoverage(filePath, mergedRanges.get(filePath));
    totalCovered += result.covered;
    totalExecutable += result.executable;

    const ratio = result.executable === 0 ? 100 : (result.covered / result.executable) * 100;
    const rel = path.relative(root, filePath);
    console.log(`${ratio.toFixed(2).padStart(6)}%  ${result.covered}/${result.executable}  ${rel}`);

    if (result.covered !== result.executable) {
      hasGap = true;
      console.log('        uncovered lines: ' + result.uncoveredLines.join(', '));
    }
  }

  const totalRatio = totalExecutable === 0 ? 100 : (totalCovered / totalExecutable) * 100;
  console.log(`Total: ${totalRatio.toFixed(2)}% (${totalCovered}/${totalExecutable})`);

  if (hasGap || totalCovered !== totalExecutable) {
    process.exit(1);
  }
}

runTestsWithCoverage();
try {
  reportAndValidate();
} finally {
  fs.rmSync(coverageDir, { recursive: true, force: true });
}
