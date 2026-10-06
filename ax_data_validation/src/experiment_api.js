/**
 * experiment_api.js
 * Client-side API layer for fetching exhaustive experiment catalog, metadata, comparison tables, and predictions.
 */

const CACHE = {
  catalog: null,
  metadata: new Map(),
  campaignCatalog: null,
  comparison: new Map(),
  predictions: new Map(),
  featureSelections: new Map(),
};

let staticCatalog = null;
const staticIndexes = new Map();

// The local Python API serves the original stored-code campaign only.
// Other campaigns, including the dashboard default quick screening, use their
// own static index declared in campaign_catalog.json.
const DEFAULT_CAMPAIGN = 'stored_control_codes';

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
      const res = await fetch('./cache/combination_catalog.json', { cache: 'no-store' });
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

export async function fetchCampaignCatalog(force = false) {
  if (!force && CACHE.campaignCatalog) return CACHE.campaignCatalog;
  try {
    const res = await fetch('./cache/campaign_catalog.json', { cache: 'no-store' });
    if (res.ok) {
      CACHE.campaignCatalog = await res.json();
      return CACHE.campaignCatalog;
    }
  } catch (err) {
    console.warn('[ExperimentAPI] Campaign catalog unavailable; using baseline only.', err);
  }
  CACHE.campaignCatalog = {
    default_campaign: DEFAULT_CAMPAIGN,
    campaigns: [{
      id: DEFAULT_CAMPAIGN,
      name: '시설별 제어코드 기준',
      description: 'DB에 저장된 시설 제어 코드를 구분한 최초 전수실험',
      status: 'completed', progress_percent: 100,
      data_path: './cache/experiment_index.json',
      predictions_path: './cache/predictions',
      feature_selections_path: './cache/feature_selections'
    }]
  };
  return CACHE.campaignCatalog;
}

async function campaignDefinition(campaign = DEFAULT_CAMPAIGN) {
  const catalog = await fetchCampaignCatalog();
  return catalog.campaigns?.find(item => item.id === campaign)
    || catalog.campaigns?.find(item => item.id === catalog.default_campaign)
    || catalog.campaigns?.[0];
}

async function getStaticIndex(campaign = DEFAULT_CAMPAIGN, force = false) {
  if (!force && staticIndexes.has(campaign)) return staticIndexes.get(campaign);
  const definition = await campaignDefinition(campaign);
  const path = definition?.data_path || './cache/experiment_index.json';
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Static index HTTP error ${res.status}: ${path}`);
  const index = await res.json();
  staticIndexes.set(campaign, index);
  return index;
}

export async function fetchMetadata(force = false, campaign = DEFAULT_CAMPAIGN) {
  if (!force && CACHE.metadata.has(campaign)) return CACHE.metadata.get(campaign);
  try {
    if (campaign !== DEFAULT_CAMPAIGN) throw new Error('campaign-specific static metadata');
    const res = await fetch('./api/experiments/metadata');
    if (res.ok) {
      const data = await res.json();
      CACHE.metadata.set(campaign, data);
      return data;
    }
  } catch (err) {
    if (campaign === DEFAULT_CAMPAIGN) console.warn('[ExperimentAPI] API metadata failed, falling back to static index:', err);
  }

  // Fallback to static cache/experiment_index.json
  try {
    const index = await getStaticIndex(campaign, force);
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
    const campaignMeta = index.campaign || {
        id: "exhaustive_e1_e7_v1",
        name: "8개 타깃·6개 모델·127개 변수군 조합 전수실험",
        total_combinations: 127,
        total_targets: 8,
        total_models: 6,
        seeds: [42, 52, 62]
      };
    const campaignSeeds = Array.isArray(campaignMeta.seeds) && campaignMeta.seeds.length
      ? campaignMeta.seeds
      : [42, 52, 62];
    const data = {
      campaign: campaignMeta,
      targets: targets_meta,
      models: ["poisson", "random_forest", "catboost", "mlp", "tabm", "tft"],
      seeds: campaignSeeds
    };
    CACHE.metadata.set(campaign, data);
    return data;
  } catch (err) {
    console.error('[ExperimentAPI] Static metadata fallback failed:', err);
    throw err;
  }
}

export async function fetchFeatureSelection(target, group, force = false, campaign = DEFAULT_CAMPAIGN) {
  const definition = await campaignDefinition(campaign);
  const root = definition?.feature_selections_path || './cache/feature_selections';
  if (definition?.feature_selection_layout === 'by_combination') {
    const cacheKey = `${campaign}::${target}::${group}`;
    if (!force && CACHE.featureSelections.has(cacheKey)) {
      return CACHE.featureSelections.get(cacheKey);
    }
    const res = await fetch(`${root}/${encodeURIComponent(target)}/${encodeURIComponent(group)}.json`, { cache: 'no-store' });
    if (!res.ok) throw new Error(`Feature selection detail HTTP error ${res.status}`);
    const data = await res.json();
    CACHE.featureSelections.set(cacheKey, data);
    return data;
  }
  const cacheKey = `${campaign}::${target}`;
  if (!force && CACHE.featureSelections.has(cacheKey)) {
    return CACHE.featureSelections.get(cacheKey)?.combinations?.[group] || null;
  }
  const res = await fetch(`${root}/${encodeURIComponent(target)}.json`, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Feature selection detail HTTP error ${res.status}`);
  const data = await res.json();
  CACHE.featureSelections.set(cacheKey, data);
  return data?.combinations?.[group] || null;
}

export async function fetchComparison(target, model, split, force = false, campaign = DEFAULT_CAMPAIGN) {
  const cacheKey = `${campaign}::${target}::${model}::${split}`;
  if (!force && CACHE.comparison.has(cacheKey)) {
    return CACHE.comparison.get(cacheKey);
  }
  try {
    if (campaign !== DEFAULT_CAMPAIGN) throw new Error('campaign-specific static comparison');
    const url = `./api/experiments/comparison?target=${encodeURIComponent(target)}&model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      CACHE.comparison.set(cacheKey, data);
      return data;
    }
  } catch (err) {
    if (campaign === DEFAULT_CAMPAIGN) console.warn('[ExperimentAPI] API comparison failed, falling back to static calculation:', err);
  }

  // Fallback to computing comparison from static catalog & index
  try {
    const catalog = await fetchCatalog();
    const index = await getStaticIndex(campaign, force);
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
          status: ginfo.campaign_role === "not_in_scope" ? "not_in_scope" : (split === "test" ? "not_evaluated" : "unstarted"),
          reason: ginfo.campaign_role === "not_in_scope" ? "E6 의미 통합 재실험 범위 외" : null,
          source: ginfo.campaign_role || null,
          source_label: null,
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
          feature_count: res_item.feature_count ?? combo.feature_count,
          category_names: combo.group_names,
          category_ids: combo.groups,
          features: combo.features,
          completed_seeds: res_item.completed_seeds || 0,
          status: res_item.status || "unstarted",
          reason: res_item.reason || null,
          source: res_item.source || null,
          source_label: res_item.source_label || null,
          is_winner: (gid === winner_group),
          params: res_item.params || {},
          best_epoch: res_item.best_epoch,
          n_eval: res_item.n_eval || 0,
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
      campaign,
      campaign_meta: index.campaign || {},
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

export async function fetchPredictions(target, group, model, split, seed = 'all', force = false, campaign = DEFAULT_CAMPAIGN) {
  const cacheKey = `${campaign}::${target}::${group}::${model}::${split}::${seed}`;
  if (!force && CACHE.predictions.has(cacheKey)) {
    return CACHE.predictions.get(cacheKey);
  }
  try {
    if (campaign !== DEFAULT_CAMPAIGN) throw new Error('campaign-specific static predictions');
    const url = `./api/experiments/predictions?target=${encodeURIComponent(target)}&group=${encodeURIComponent(group)}&model=${encodeURIComponent(model)}&split=${encodeURIComponent(split)}&seed=${encodeURIComponent(seed)}`;
    const res = await fetch(url);
    if (res.ok) {
      const data = await res.json();
      CACHE.predictions.set(cacheKey, data);
      return data;
    }
  } catch (err) {
    if (campaign === DEFAULT_CAMPAIGN) console.warn('[ExperimentAPI] API predictions failed, falling back to static predictions cache:', err);
  }

  // Fallback to static cache files
  try {
    const index = await getStaticIndex(campaign, force);
    const t_data = index.targets?.[target] || {};
    const key = `${group}::${model}::${split}`;
    const res_entry = t_data.results?.[key];
    if (!res_entry) {
      return { status: "no_results", group_id: group, model, split, target, predictions: [], metadata: {} };
    }
    const seeds_dict = res_entry.seeds || {};
    const available_seeds = Object.keys(seeds_dict).map(s => parseInt(s, 10)).sort((a, b) => a - b);
    const definition = await campaignDefinition(campaign);
    const predictionRoot = campaign === 'semantic_control' && res_entry.source === 'reused_stored_control_codes'
      ? (definition?.baseline_predictions_path || './cache/predictions')
      : (definition?.predictions_path || './cache/predictions');

    if (String(seed).toLowerCase() === 'all' || String(seed).toLowerCase() === 'ensemble') {
      const all_pred_runs = [];
      for (const s of available_seeds) {
        const run_id = seeds_dict[String(s)]?.run_id;
        if (!run_id) continue;
        try {
          const pRes = await fetch(`${predictionRoot}/${run_id}.json`, { cache: 'no-store' });
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
      const pRes = await fetch(`${predictionRoot}/${seed_item.run_id}.json`, { cache: 'no-store' });
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

export async function refreshExperiments(campaign = DEFAULT_CAMPAIGN) {
  try {
    const res = await fetch('./api/experiments/refresh');
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    CACHE.catalog = null;
    CACHE.campaignCatalog = null;
    CACHE.metadata.clear();
    CACHE.comparison.clear();
    CACHE.predictions.clear();
    CACHE.featureSelections.clear();
    staticCatalog = null;
    staticIndexes.clear();
    return await fetchMetadata(true, campaign);
  } catch (err) {
    console.error('[ExperimentAPI] refreshExperiments failed:', err);
    throw err;
  }
}
