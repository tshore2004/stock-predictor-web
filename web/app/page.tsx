"use client";
import { useState, useEffect } from "react";

// type Matrix = number[][]; // 60×8

export default function Home() {
  const [mae, setMae] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const api = process.env.NEXT_PUBLIC_API;
  const [baseline, setBaseline] = useState<number | null>(null);
  const [ticker, setTicker] = useState("AAPL");
  // fetch accuracy once on mount
  useEffect(() => {
    fetch(`${api}/metrics`)
      .then((r) => r.json())
      .then((j) => {
        setMae(j.val_mae);
        setBaseline(j.baseline);
      });
  }, [api]);

  async function handlePredict() {

    const api = process.env.NEXT_PUBLIC_API!;
    try {
      const res = await fetch(`${api}/train-and-predict?ticker=${ticker}`);
      const { prediction } = await res.json();
      setPrediction(prediction);
    } catch (err) {
      alert("Failed to predict. Try a different ticker.");
      console.error(err);
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
        className="rounded bg-indigo-600 px-4 py-2 text-white hover:bg-indigo-700"
      >
        Predict Next Close
      </button>

      {prediction !== null && (
        <p className="text-xl">
          Model prediction:&nbsp;
          <b>${prediction.toFixed(2)}</b>
        </p>
      )}

      {mae !== null && baseline !== null && (
        <p className="text-gray-600">
          Validation MAE:&nbsp;
          <b>
            ${(mae * baseline).toFixed(2)} (
            {(mae * 100).toFixed(2)}%
            )
          </b>
        </p>
      )}
    </main>
  );
}
