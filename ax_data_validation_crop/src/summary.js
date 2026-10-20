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
