export interface Prediction {
  // main output (beat-informed)
  arrhythmia_detected: boolean
  arrhythmia_code: string
  arrhythmia_type: string
  risk_level: string

  // segment-level
  confidence: number
  source: "ml" | "rule_based" | "hybrid_rule_fallback"
  all_probabilities: Record<string, number>

  // helpful extras
  message?: string
  base_rhythm_code?: string
  main_arrhythmia_code?: string

  summary?: {
    heart_rate_bpm: number
    total_beats: number
    abnormal_beats: number
    normal_beats: number
    abnormal_percent: number
    normal_percent: number
  }

  events?: Array<{
    code: string
    label: string
    count: number
    percent?: number
  }>
}
