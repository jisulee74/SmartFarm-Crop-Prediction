/**
 * experiment_api.js
 * Client-side API layer for fetching exhaustive experiment catalog, metadata, comparison tables, and predictions.
 */

const CACHE = {
  catalog: null,
  metadata: null,
  comparison: new Map(),
  predictions: new Map(),
};

let staticCatalog = null;
let staticIndex = null;

export async function fetchCatalog(force = false) {
  if (!force && CACHE.catalog) return CACHE.catalog;
  try {
    const res = await fetch('./api/experiments/catalog');
    if (res.ok) {
      const data = await res.json();
      CACHE.catalog = data;
      return data;
    }
  } catch (err) {
    console.warn('[ExperimentAPI] API catalog fetch failed, falling back to static file:', err);
  }

  // Fallback to static cache/combination_catalog.json
  try {
    if (!staticCatalog) {
      const res = await fetch('./cache/combination_catalog.json');
      if (!res.ok) throw new Error(`Static catalog HTTP error ${res.status}`);
      staticCatalog = await res.json();
    }
    CACHE.catalog = staticCatalog;
    return staticCatalog;
  } catch (err) {
    console.error('[ExperimentAPI] Static catalog fallback failed:', err);
    throw err;
  }
}

async function getStaticIndex() {
  if (staticIndex) return staticIndex;
  const res = await fetch('./cache/experiment_index.json');
  if (!res.ok) throw new Error(`Static index HTTP error ${res.status}`);
  staticIndex = await res.json();
  return staticIndex;
}

export async function fetchMetadata(force = false) {
  if (!force && CACHE.metadata) return CACHE.metadata;
  try {
    const res = await fetch('./api/experiments/metadata');
    if (res.ok) {
      const data = await res.json();
      CACHE.metadata = data;
      return data;
    }
  } catch (err) {
    console.warn('[ExperimentAPI] API metadata failed, falling back to static index:', err);
  }

  // Fallback to static cache/experiment_index.json
  try {
    const index = await getStaticIndex();
    const targets_meta = {};
    for (const [t_id, t_data] of Object.entries(index.targets || {})) {
      targets_meta[t_id] = {
        target_id: t_id,
        crop: t_data.crop,
        crop_kr: t_data.crop_kr,
        target_kr: t_data.target_kr,
        part: t_data.part,
        status: t_data.status,
        group_count: Object.keys(t_data.groups || {}).length,
        frozen_winner: t_data.frozen?.overall,
        frozen: t_data.frozen || {},
        groups: t_data.groups || {}
      };
    }
    const data = {
      campaign: index.campaign || {
        id: "exhaustive_e1_e7_v1",
        name: "8개 타깃·6개 모델·127개 변수군 조합 전수실험",
        total_combinations: 127,
        total_targets: 8,
        total_models: 6,
        seeds: [42, 52, 62]
      },
      targets: targets_meta,
      models: ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"],
      seeds: [42, 52, 62]
    };
    CACHE.metadata = data;
    return data;
  } catch (err) {
    console.error('[ExperimentAPI] Static metadata fallback failed:', err);
    throw err;
  }
}

export async function fetchComparison(target, model, split, force = false) {
  const cacheKey = `${target}::${model}::${split}`;
  if (!force && CACHE.comparison.has(cacheKey)) {
    return CACHE.comparison.get(cacheKey);
  }
  try {
    const url = `./api/experiments/comparison?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      CACHE.comparison.set(cacheKey, data);
      return data;
    }
  } catch (err) {
    console.warn('[ExperimentAPI] API comparison failed, falling back to static calculation:', err);
  }

  // Fallback to computing comparison from static catalog & index
  try {
    const catalog = await fetchCatalog();
    const index = await getStaticIndex();
    const t_data = index.targets?.[target] || {};
    const groups_dict = t_data.groups || {};
    const results_dict = t_data.results || {};
    const frozen_winners = t_data.frozen?.winners || {};
    const winner_group = frozen_winners[model]?.group || t_data.frozen?.overall || null;

    const rows = [];
    const combos = catalog.combinations || [];

    for (const combo of combos) {
      const gid = combo.combination_id;
      const ginfo = groups_dict[gid] || combo;
      const key = `${gid}::${model}::${split}`;
      const res_item = results_dict[key];

      if (!res_item) {
        rows.push({
          group_id: gid,
          readable_name: combo.label,
          display_name: combo.display_name,
          groups: combo.groups,
          group_count: combo.group_count,
          feature_count: combo.feature_count,
          category_names: combo.group_names,
          category_ids: combo.groups,
          features: combo.features,
          completed_seeds: 0,
          status: "unstarted",
          is_winner: (gid === winner_group),
          params: {},
          bounded: {},
          raw: {}
        });
      } else {
        rows.push({
          group_id: gid,
          readable_name: combo.label,
          display_name: combo.display_name,
          groups: combo.groups,
          group_count: combo.group_count,
          feature_count: combo.feature_count,
          category_names: combo.group_names,
          category_ids: combo.groups,
          features: combo.features,
          completed_seeds: res_item.completed_seeds || 0,
          status: res_item.status || "unstarted",
          is_winner: (gid === winner_group),
          params: res_item.params || {},
          best_epoch: res_item.best_epoch,
          bounded: res_item.bounded || {},
          raw: res_item.raw || {}
        });
      }
    }

    // Sort: completed with valid RMSE first (ascending), then unstarted
    rows.sort((a, b) => {
      const aRmse = a.bounded?.rmse_mean;
      const bRmse = b.bounded?.rmse_mean;
      if (aRmse != null && bRmse != null) {
        if (aRmse !== bRmse) return aRmse - bRmse;
        const aMae = a.bounded?.mae_mean ?? 999999;
        const bMae = b.bounded?.mae_mean ?? 999999;
        if (aMae !== bMae) return aMae - bMae;
        return (a.feature_count || 999) - (b.feature_count || 999);
      }
      if (aRmse != null) return -1;
      if (bRmse != null) return 1;
      // Both unstarted: sort by group_count asc, then label lexical
      if (a.group_count !== b.group_count) return a.group_count - b.group_count;
      return a.readable_name.localeCompare(b.readable_name);
    });

    let rankCounter = 1;
    rows.forEach((r) => {
      if (r.bounded?.rmse_mean != null) {
        r.rank = rankCounter++;
      } else {
        r.rank = null;
      }
    });

    const provisional_winner = rows.length > 0 && rows[0].bounded?.rmse_mean != null ? rows[0].group_id : null;
    const data = {
      target,
      model,
      split,
      target_status: t_data.status || "pending_experiment",
      frozen_winner: winner_group,
      provisional_winner,
      total_candidates: rows.length,
      rows
    };
    CACHE.comparison.set(cacheKey, data);
    return data;
  } catch (err) {
    console.error('[ExperimentAPI] Static comparison calculation failed:', err);
    throw err;
  }
}

export async function fetchPredictions(target, group, model, split, seed = 'all', force = false) {
  const cacheKey = `${target}::${group}::${model}::${split}::${seed}`;
  if (!force && CACHE.predictions.has(cacheKey)) {
    return CACHE.predictions.get(cacheKey);
  }
  try {
    const url = `./api/experiments/predictions?target=${encodeURIComponent(target)}&group=${encodeURIComponent(group)}&model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}&seed=${encodeURIComponent(seed)}`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      CACHE.predictions.set(cacheKey, data);
      return data;
    }
  } catch (err) {
    console.warn('[ExperimentAPI] API predictions failed, falling back to static predictions cache:', err);
  }

  // Fallback to static cache files
  try {
    const index = await getStaticIndex();
    const t_data = index.targets?.[target] || {};
    const key = `${group}::${model}::${split}`;
    const res_entry = t_data.results?.[key];
    if (!res_entry) {
      return { status: "no_results", group_id: group, model, split, target, predictions: [], metadata: {} };
    }
    const seeds_dict = res_entry.seeds || {};
    const available_seeds = Object.keys(seeds_dict).map(s => parseInt(s, 10)).sort((a, b) => a - b);

    if (String(seed).toLowerCase() === 'all' || String(seed).toLowerCase() === 'ensemble') {
      const all_pred_runs = [];
      for (const s of available_seeds) {
        const run_id = seeds_dict[String(s)]?.run_id;
        if (!run_id) continue;
        try {
          const pRes = await fetch(`./cache/predictions/${run_id}.json`);
          if (pRes.ok) {
            const pJson = await pRes.json();
            if (pJson.predictions && pJson.predictions.length > 0) {
              all_pred_runs.push(pJson.predictions);
            }
          }
        } catch (e) {}
      }
      if (all_pred_runs.length === 0) {
        return { status: "no_predictions", predictions: [], metadata: res_entry };
      }
      const n_rows = all_pred_runs[0].length;
      const ensemble_preds = [];
      for (let i = 0; i < n_rows; i++) {
        const row0 = all_pred_runs[0][i];
        const raw_vals = all_pred_runs.map(run => run[i]?.prediction_raw).filter(v => v != null);
        const bnd_vals = all_pred_runs.map(run => run[i]?.prediction_bounded).filter(v => v != null);
        const mean = arr => arr.reduce((acc, c) => acc + c, 0) / arr.length;
        ensemble_preds.push({
          facility_id: row0.facility_id,
          crop_sn: String(row0.crop_sn),
          sample_num: String(row0.sample_num),
          row_id: row0.row_id,
          feature_date: row0.feature_date,
          target_date: row0.target_date,
          target: row0.target,
          prediction_raw: raw_vals.length > 0 ? mean(raw_vals) : null,
          prediction_bounded: bnd_vals.length > 0 ? mean(bnd_vals) : null,
          upper_bound: row0.upper_bound,
          seeds_averaged: raw_vals.length
        });
      }
      const data = {
        status: "success",
        mode: "ensemble",
        seeds_count: all_pred_runs.length,
        available_seeds,
        target,
        group_id: group,
        model,
        split,
        metadata: res_entry,
        predictions: ensemble_preds
      };
      CACHE.predictions.set(cacheKey, data);
      return data;
    } else {
      const s_int = parseInt(seed, 10);
      const seed_item = seeds_dict[String(s_int)];
      if (!seed_item || !seed_item.run_id) {
        return { status: "seed_not_found", available_seeds, predictions: [], metadata: res_entry };
      }
      const pRes = await fetch(`./cache/predictions/${seed_item.run_id}.json`);
      if (!pRes.ok) throw new Error(`Prediction fetch HTTP error ${pRes.status}`);
      const pData = await pRes.json();
      const preds = (pData.predictions || []).map(r => ({
        ...r,
        crop_sn: String(r.crop_sn),
        sample_num: String(r.sample_num)
      }));
      const data = {
        status: "success",
        mode: "single_seed",
        seed: s_int,
        run_id: seed_item.run_id,
        available_seeds,
        target,
        group_id: group,
        model,
        split,
        metadata: res_entry,
        predictions: preds
      };
      CACHE.predictions.set(cacheKey, data);
      return data;
    }
  } catch (err) {
    console.error('[ExperimentAPI] Static predictions fallback failed:', err);
    throw err;
  }
}

export async function refreshExperiments() {
  try {
    const res = await fetch('./api/experiments/refresh');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    CACHE.catalog = null;
    CACHE.metadata = null;
    CACHE.comparison.clear();
    CACHE.predictions.clear();
    staticCatalog = null;
    staticIndex = null;
    return await fetchMetadata(true);
  } catch (err) {
    console.error('[ExperimentAPI] refreshExperiments failed:', err);
    throw err;
  }
}
