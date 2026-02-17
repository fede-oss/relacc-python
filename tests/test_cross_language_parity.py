import json
import subprocess
import sys
from pathlib import Path

import pytest

from relacc import relacc as PyRelAcc
from relacc.geom.point import Point
from relacc.gestures.ptaligntype import PtAlignType


ROOT = Path(__file__).resolve().parents[1]
JS_LIB = ROOT / "lib" / "relacc.js"


def p(x, y, t, sid):
    return Point(x, y, t, sid)


def _python_metrics():
    summaryPts = [p(0, 0, 0, 0), p(2, 2, 10, 0), p(4, 0, 20, 1)]
    chronoPts = [p(0, 0, 0, 0), p(2, 1, 10, 0), p(4, 1, 20, 1)]
    cloudPts = [p(0, 0, 0, 0), p(2, 2, 10, 0), p(4, 0, 20, 1)]

    gesture = type("GestureObj", (), {"points": chronoPts})()

    class SummaryShape:
        points = summaryPts

        @staticmethod
        def getPoints():
            return summaryPts

        @staticmethod
        def alignGesture(_, alignmentType=None):
            if alignmentType == PtAlignType.CLOUD_MATCH:
                return cloudPts
            return chronoPts

    summaryShape = SummaryShape()

    return {
        "shapeError": PyRelAcc.shapeError(gesture, summaryShape),
        "shapeVariability": PyRelAcc.shapeVariability(gesture, summaryShape),
        "lengthError": PyRelAcc.lengthError(gesture, summaryShape),
        "sizeError": PyRelAcc.sizeError(gesture, summaryShape),
        "bendingError": PyRelAcc.bendingError(gesture, summaryShape),
        "bendingVariability": PyRelAcc.bendingVariability(gesture, summaryShape),
        "timeError": PyRelAcc.timeError(gesture, summaryShape),
        "timeVariability": PyRelAcc.timeVariability(gesture, summaryShape),
        "velocityError": PyRelAcc.velocityError(gesture, summaryShape),
        "velocityVariability": PyRelAcc.velocityVariability(gesture, summaryShape),
        "strokeError": PyRelAcc.strokeError(gesture, summaryShape),
        "strokeOrderError": PyRelAcc.strokeOrderError(gesture, summaryShape),
    }


def _js_metrics():
    if not JS_LIB.exists():
        pytest.skip("JS reference implementation is not present in Python-only repository.")
    script = r'''
const RelAcc = require('./lib/relacc');
const Point = require('./lib/geom/point');
const PtAlignType = require('./lib/gestures/ptaligntype');
function p(x,y,t,sid){ return new Point(x,y,t,sid); }
const summaryPts = [p(0,0,0,0), p(2,2,10,0), p(4,0,20,1)];
const chronoPts  = [p(0,0,0,0), p(2,1,10,0), p(4,1,20,1)];
const cloudPts   = [p(0,0,0,0), p(2,2,10,0), p(4,0,20,1)];
const gesture = { points: chronoPts };
const summaryShape = {
  points: summaryPts,
  getPoints: function(){ return summaryPts; },
  alignGesture: function(_, alignmentType){
    if (alignmentType === PtAlignType.CLOUD_MATCH) return cloudPts;
    return chronoPts;
  }
};
const out = {
  shapeError: RelAcc.shapeError(gesture, summaryShape),
  shapeVariability: RelAcc.shapeVariability(gesture, summaryShape),
  lengthError: RelAcc.lengthError(gesture, summaryShape),
  sizeError: RelAcc.sizeError(gesture, summaryShape),
  bendingError: RelAcc.bendingError(gesture, summaryShape),
  bendingVariability: RelAcc.bendingVariability(gesture, summaryShape),
  timeError: RelAcc.timeError(gesture, summaryShape),
  timeVariability: RelAcc.timeVariability(gesture, summaryShape),
  velocityError: RelAcc.velocityError(gesture, summaryShape),
  velocityVariability: RelAcc.velocityVariability(gesture, summaryShape),
  strokeError: RelAcc.strokeError(gesture, summaryShape),
  strokeOrderError: RelAcc.strokeOrderError(gesture, summaryShape)
};
console.log(JSON.stringify(out));
'''
    res = subprocess.run(["node", "-e", script], cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(res.stdout)


def test_python_matches_js_metrics():
    py = _python_metrics()
    js = _js_metrics()

    for key in js:
        assert py[key] == pytest.approx(js[key], abs=1e-9)
