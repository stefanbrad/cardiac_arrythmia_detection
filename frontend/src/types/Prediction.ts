export interface Prediction {
  arrhythmia_detected: boolean
  arrhythmia_code: string
  arrhythmia_type: string
  risk_level: string

  confidence: number
  source: "ml" | "rule_based" | "hybrid_rule_fallback"
  all_probabilities: Record<string, number>

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
