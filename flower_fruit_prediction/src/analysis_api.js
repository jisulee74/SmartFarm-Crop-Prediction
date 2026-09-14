/**
 * analysis_api.js
 * Client-side API fetcher for Growth Prediction & AX Direction Analysis data
 */

let cachedBundle = null;

export async function fetchAnalysisBundle() {
  if (cachedBundle) {
    return cachedBundle;
  }
  try {
    const response = await fetch('./api/analysis/bundle', {
      headers: { 'Connection': 'close' }
    });
    if (response.ok) {
      cachedBundle = await response.json();
      return cachedBundle;
    }
  } catch (err) {
    console.warn('[AnalysisAPI] Server API unavailable, attempting static fallback:', err);
  }

  // Fallback for static GitHub Pages / file hosting
  try {
    const staticRes = await fetch('./ax_direction/dashboard_bundle.json');
    if (staticRes.ok) {
      cachedBundle = await staticRes.json();
      return cachedBundle;
    }
  } catch (err) {
    console.error('[AnalysisAPI] Static fallback failed:', err);
  }

  throw new Error('Failed to load analysis bundle from API and static fallback');
}

export function clearAnalysisCache() {
  cachedBundle = null;
}
