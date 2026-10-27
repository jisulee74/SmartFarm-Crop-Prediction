// The summary uses frozen selected-model Test metrics, independent of advanced selectors.
const buttons = document.querySelectorAll('[data-summary-crop]');
buttons.forEach(button => button.addEventListener('click', () => {
  const crop = button.dataset.summaryCrop;
  buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  document.querySelectorAll('[data-summary-panel]').forEach(panel => {
    panel.hidden = panel.dataset.summaryPanel !== crop;
  });
  const advancedCrop = document.querySelector(`input[name="exp_crop_radio"][value="${crop}"]`);
  if (advancedCrop && !advancedCrop.checked) {
    advancedCrop.checked = true;
    advancedCrop.dispatchEvent(new Event('change', { bubbles: true }));
  }
}));
document.getElementById('crop_advanced')?.addEventListener('toggle', event => {
  if (event.target.open) requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
});

// Load the immutable publication summary relative to this module, not the shared page <base>.
import { renderPooled, showUnavailable } from './presentation.js?v=20261027';
const summaryRoot = document.querySelector('.crop-summary');
if (summaryRoot) {
  try {
    const response = await fetch(new URL('../summary.json', import.meta.url), { cache: 'no-store' });
    if (!response.ok) throw new Error('검증된 결과 파일을 불러오지 못했습니다.');
    const bundle = await response.json();
    renderPooled(summaryRoot, bundle);
    document.querySelectorAll('[data-current-result-stage]').forEach(node => {
      node.textContent = `${bundle.seed_count}시드 결과와 구분한 발표자료 기준 목표`;
    });
  } catch (error) {
    showUnavailable(summaryRoot, '검증된 결과 파일을 불러오지 못했습니다.');
  }
}
