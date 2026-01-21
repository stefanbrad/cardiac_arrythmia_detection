import type { Prediction } from "../types/Prediction"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { GlowingEffect } from "@/components/ui/glowing-effect"
import { FileText, AlertTriangle } from "lucide-react"

interface Props {
  prediction: Prediction
}

export function ResultsDisplay({ prediction }: Props) {
  const probs = prediction.all_probabilities ?? {}
  const sortedProbs = Object.entries(probs).sort((a, b) => b[1] - a[1])

  const summary = prediction.summary
  const events = prediction.events ?? []

  const confidencePct = Math.round((prediction.confidence ?? 0) * 100)

  const normalPct = summary ? Math.round((summary.normal_percent ?? 0) * 100) : 0
  const abnormalPct = summary ? Math.round((summary.abnormal_percent ?? 0) * 100) : 0

  const totalBeats = summary?.total_beats ?? 0
  const normalBeats = summary?.normal_beats ?? 0
  const abnormalBeats = summary?.abnormal_beats ?? 0
  const heartRate = summary?.heart_rate_bpm ?? 0

  // Useful warning if beat stats are based on too few beats
  const showFewBeatsWarning = totalBeats > 0 && totalBeats < 20

  return (
    <section id="results-section" className="py-20">
      <div className="container mx-auto px-4 max-w-3xl">
        <Card className="relative overflow-hidden">
          <GlowingEffect
            spread={40}
            glow={true}
            disabled={false}
            proximity={64}
            inactiveZone={0.01}
            borderWidth={2}
          />

          <CardHeader className="text-center">
            <div className="flex items-center justify-center gap-2">
              <FileText className="h-6 w-6 text-red-500" />
              <CardTitle className="text-2xl">Analysis Results</CardTitle>
            </div>

            <CardDescription>
              Summary and probabilities from the analyzed ECG window(s)
            </CardDescription>
          </CardHeader>

          <CardContent>
            {/* Top summary */}
            <div className="space-y-2">
              <p className="text-lg">
                <strong>Arrhythmia detected:</strong>{" "}
                {prediction.arrhythmia_detected ? prediction.arrhythmia_code : "No"}
              </p>

              <p className="text-lg">
                <strong>Type:</strong> {prediction.arrhythmia_type}
              </p>

              <p className="text-lg">
                <strong>Risk level:</strong> {prediction.risk_level}
              </p>

              <p className="text-lg">
                <strong>Confidence:</strong> {confidencePct}%
              </p>

              {prediction.message && (
                <p className="text-sm text-gray-600 dark:text-gray-400">{prediction.message}</p>
              )}

              {showFewBeatsWarning && (
                <div className="mt-3 flex items-start gap-2 rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800 dark:border-yellow-900/50 dark:bg-yellow-950/30 dark:text-yellow-200">
                  <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <div>
                    <strong>Warning:</strong> Only {totalBeats} beats were detected in the analyzed
                    window(s). Beat percentages may be unreliable.
                  </div>
                </div>
              )}
            </div>

            {/* Beat-based cards */}
            {summary && (
              <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="p-4 rounded-xl border">
                  <div className="text-3xl font-bold">{normalPct}%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Normal Beats (beat-based)
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {normalBeats} / {totalBeats}
                  </div>
                </div>

                <div className="p-4 rounded-xl border">
                  <div className="text-3xl font-bold">{abnormalPct}%</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">
                    Abnormal Beats (beat-based)
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {abnormalBeats} / {totalBeats}
                  </div>
                </div>

                <div className="p-4 rounded-xl border">
                  <div className="text-3xl font-bold">{heartRate}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400">Heart Rate (bpm)</div>
                </div>
              </div>
            )}

            {/* Events */}
            <div className="mt-8">
              <h3 className="font-semibold mb-2">Detected Arrhythmias (beat events)</h3>

              {events.length === 0 ? (
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  No arrhythmias detected.
                </p>
              ) : (
                <ul className="space-y-2">
                  {events.map((e) => (
                    <li
                      key={e.code}
                      className="flex justify-between items-center border rounded-lg p-3"
                    >
                      <span className="flex items-center gap-2">
                        <strong>{e.code}</strong>
                        {"percent" in e && typeof (e as any).percent === "number" ? (
                          <span className="text-xs text-gray-500">
                            ({Math.round(((e as any).percent as number) * 100)}%)
                          </span>
                        ) : null}
                      </span>
                      <span className="text-sm text-gray-700 dark:text-gray-300">
                        {e.count} events
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            {/* Probabilities */}
            {sortedProbs.length > 0 && (
              <div className="mt-8">
                <h3 className="font-semibold mb-1">Class probabilities (segment-level)</h3>
                <p className="text-xs text-gray-500 mb-3">
                  These probabilities are computed on the whole analyzed window(s), not on
                  individual beats.
                </p>

                <div className="space-y-1">
                  {sortedProbs.map(([label, value]) => (
                    <div key={label} className="flex justify-between text-sm">
                      <span>{label}</span>
                      <span>{Math.round(value * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </section>
  )
}
