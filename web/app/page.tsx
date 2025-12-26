"use client";
import { useState, useEffect } from "react";

// type Matrix = number[][]; // 60×8

export default function Home() {
  const [mae, setMae] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [modelSource, setModelSource] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [baseline, setBaseline] = useState<number | null>(null);
  const [ticker, setTicker] = useState("AAPL");
  const [hitRate, setHitRate] = useState<number | null>(null);
  const [rSquared, setRSquared] = useState<number | null>(null);
  const [correlation, setCorrelation] = useState<number | null>(null);
  const [accuracyWithin2Pct, setAccuracyWithin2Pct] = useState<number | null>(null);
  const [mape, setMape] = useState<number | null>(null);
  const [rmse, setRmse] = useState<number | null>(null);
  
  // Get API URL - ensure it's consistent
  const api = typeof window !== 'undefined' 
    ? (process.env.NEXT_PUBLIC_API || "http://localhost:8000")
    : "http://localhost:8000"; // Server-side fallback
  
  // fetch accuracy once on mount
  useEffect(() => {
    fetch(`${api}/metrics?ticker=${ticker}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (!j) return;
        setMae(j.val_mae);
        setBaseline(j.baseline);
        setHitRate(j.hit_rate ?? null);
        setRSquared(j.r_squared ?? null);
        setCorrelation(j.correlation ?? null);
        setAccuracyWithin2Pct(j.accuracy_within_2pct ?? null);
        setMape(j.mape ?? null);
        setRmse(j.val_rmse ?? null);
      })
      .catch((err) => {
        console.error("Failed to fetch metrics:", err);
      });
  }, [api, ticker]);

  async function handlePredict() {
    setIsLoading(true);
    setPrediction(null);
    setModelSource(null);
    
    try {
      const res = await fetch(`${api}/predict?ticker=${ticker}`);
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      const data = await res.json();
      setPrediction(data.prediction);
      setModelSource(data.model_source);
      
      // Refresh metrics after prediction (in case model was retrained)
      fetch(`${api}/metrics?ticker=${ticker}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((j) => {
          if (j) {
            setMae(j.val_mae);
            setBaseline(j.baseline);
            setHitRate(j.hit_rate ?? null);
            setRSquared(j.r_squared ?? null);
            setCorrelation(j.correlation ?? null);
            setAccuracyWithin2Pct(j.accuracy_within_2pct ?? null);
            setMape(j.mape ?? null);
            setRmse(j.val_rmse ?? null);
          }
        });
    } catch (err) {
      console.error("Prediction error:", err);
      alert(`Failed to predict: ${err instanceof Error ? err.message : 'Unknown error'}. Make sure the API server is running on ${api}`);
    } finally {
      setIsLoading(false);
    }
  }

  // function fmtPct(dollars: number, base: number) {
  //   return ((dollars / base) * 100).toFixed(2);
  // }

  return (
    <main className="min-h-screen flex flex-col items-center gap-6 p-6">
      <h1 className="text-3xl font-bold">Stock Predictor Demo</h1>
      <div className="flex gap-2 items-center">
        <input
          className="border px-3 py-2 rounded text-lg"
          type="text"
          placeholder="Enter Ticker (e.g. AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value.toUpperCase())}
        />
      </div>
      <button
        onClick={handlePredict}
        disabled={isLoading}
        className="rounded bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
      >
        {isLoading && (
          <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        )}
        {isLoading ? "Training/Predicting..." : "Predict Next Close"}
      </button>

      {isLoading && (
        <div className="flex flex-col items-center gap-2">
          <p className="text-lg text-gray-600">Training model... This may take a minute.</p>
        </div>
      )}

      {prediction !== null && (
        <div className="flex flex-col items-center gap-2">
          <p className="text-xl">
            Model prediction:&nbsp;
            <b>${prediction.toFixed(2)}</b>
          </p>
          {modelSource && (
            <p className="text-sm text-gray-500">
              Model status:&nbsp;
              <span className={`font-semibold ${
                modelSource === "trained" 
                  ? "text-green-600" 
                  : "text-blue-600"
              }`}>
                {modelSource === "trained" 
                  ? "✓ Trained" 
                  : "✓ Loaded from cache"}
              </span>
            </p>
          )}
        </div>
      )}

      {/* Model Performance Metrics Section */}
      <div className="w-full max-w-2xl mt-8 p-6 bg-gray-50 rounded-lg shadow">
        <h2 className="text-2xl font-bold mb-4 text-center">Model Performance Metrics</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Directional Accuracy */}
          {hitRate !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Directional Accuracy</p>
              <p className="text-2xl font-bold text-green-600">
                {(hitRate * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">Correct up/down predictions</p>
            </div>
          )}

          {/* R² Score */}
          {rSquared !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">R² Score</p>
              <p className="text-2xl font-bold text-blue-600">
                {rSquared.toFixed(3)}
              </p>
              <p className="text-xs text-gray-500 mt-1">Variance explained</p>
            </div>
          )}

          {/* Correlation */}
          {correlation !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Correlation</p>
              <p className="text-2xl font-bold text-purple-600">
                {correlation.toFixed(3)}
              </p>
              <p className="text-xs text-gray-500 mt-1">With actual returns</p>
            </div>
          )}

          {/* Accuracy within 2% */}
          {accuracyWithin2Pct !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Within 2% Threshold</p>
              <p className="text-2xl font-bold text-indigo-600">
                {(accuracyWithin2Pct * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">Predictions within 2%</p>
            </div>
          )}

          {/* MAE */}
          {mae !== null && baseline !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Mean Absolute Error</p>
              <p className="text-2xl font-bold text-orange-600">
                ${(mae * baseline).toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                ({(mae * 100).toFixed(2)}% of price)
              </p>
            </div>
          )}

          {/* RMSE */}
          {rmse !== null && baseline !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Root Mean Squared Error</p>
              <p className="text-2xl font-bold text-red-600">
                ${(rmse * baseline).toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                ({(rmse * 100).toFixed(2)}% of price)
              </p>
            </div>
          )}

          {/* MAPE */}
          {mape !== null && (
            <div className="p-4 bg-white rounded border">
              <p className="text-sm text-gray-600">Mean Absolute % Error</p>
              <p className="text-2xl font-bold text-teal-600">
                {mape.toFixed(2)}%
              </p>
              <p className="text-xs text-gray-500 mt-1">Average percentage error</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
