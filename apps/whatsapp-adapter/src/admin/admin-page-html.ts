/** Página única do painel admin (sem bundler). */
export const ADMIN_SINGLE_PAGE_HTML = `<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Admin — Política WhatsApp Finance</title>
  <style>
    :root { font-family: system-ui, sans-serif; background: #111; color: #eee; }
    body { max-width: 960px; margin: 0 auto; padding: 1.5rem; }
    h1 { font-size: 1.25rem; }
    section { margin: 1.5rem 0; padding: 1rem; background: #1a1a1a; border-radius: 8px; }
    label { display: block; margin: 0.35rem 0; }
    input, textarea, select, button { font: inherit; padding: 0.4rem 0.5rem; border-radius: 4px; border: 1px solid #444; background: #222; color: #eee; }
    textarea { width: 100%; min-height: 4rem; box-sizing: border-box; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { border-bottom: 1px solid #333; padding: 0.35rem; text-align: left; vertical-align: middle; }
    .row { display: flex; flex-wrap: wrap; gap: 0.75rem; align-items: flex-end; }
    .err { color: #f88; margin-top: 0.5rem; }
    .ok { color: #8f8; margin-top: 0.5rem; }
    code { font-size: 0.85rem; }
  </style>
</head>
<body>
  <h1>Política de acesso</h1>
  <p>Envie o header <code>Authorization: Bearer …</code> (mesmo valor de <code>ADMIN_TOKEN</code>).</p>
  <section>
    <label>Token
      <input type="password" id="token" placeholder="ADMIN_TOKEN" style="width:100%;max-width:420px"/>
    </label>
    <button type="button" id="saveTok">Guardar na sessão</button>
    <p class="ok" id="tokStat"></p>
  </section>
  <section>
    <h2>Identidades</h2>
    <div class="row">
      <label>Número (dígitos) <input id="newId" placeholder="5511999999999"/></label>
      <label>Nome <input id="newLabel" placeholder="opcional"/></label>
      <label>Modelo <select id="newModel"></select></label>
      <button type="button" id="btnAdd">Adicionar</button>
      <button type="button" id="btnReload">Invalidar cache do bot</button>
    </div>
    <p class="err" id="idErr"></p>
    <div id="idTable"></div>
  </section>
  <section>
    <h2>Mensagens</h2>
    <div id="tpls"></div>
  </section>
  <script>
  const $ = (id) => document.getElementById(id);
  const tokKey = 'admin_bearer';
  function bearer() { return sessionStorage.getItem(tokKey) || $('token').value.trim(); }
  $('saveTok').onclick = () => {
    sessionStorage.setItem(tokKey, $('token').value.trim());
    $('tokStat').textContent = 'Token guardado na sessionStorage.';
  };
  let MODELS = [];
  async function api(path, opts) {
    const headers = Object.assign({}, (opts && opts.headers) || {}, {
      'Authorization': 'Bearer ' + bearer(),
      'Content-Type': 'application/json'
    });
    const r = await fetch(path, Object.assign({}, opts || {}, { headers }));
    const t = await r.text();
    let j = null;
    try { j = JSON.parse(t); } catch (_) {}
    if (!r.ok) throw new Error((j && j.error) || t || r.status);
    return j;
  }
  function modelOptions(selected) {
    return MODELS.map(function(m) {
      return '<option value="' + m.replace(/"/g,'&quot;') + '"' + (m === selected ? ' selected' : '') + '>' + m.replace(/</g,'&lt;') + '</option>';
    }).join('');
  }
  async function loadModels() {
    const j = await api('/api/models');
    MODELS = j.models || [];
    $('newModel').innerHTML = modelOptions(MODELS[0] || '');
  }
  async function loadIdentities() {
    const rows = await api('/api/identities');
    const featCat = await api('/api/feature-catalog');
    const feats = featCat.features || [];
    var head = '<tr><th>Número</th><th>Nome</th><th>Ativo</th><th>Modelo</th>';
    feats.forEach(function(f) { head += '<th>' + f.key.replace(/</g,'&lt;') + '</th>'; });
    head += '<th></th></tr>';
    var body = rows.map(function(r) {
      var tr = '<tr data-id="' + r.contact_digits + '">';
      tr += '<td>' + r.contact_digits + '</td>';
      tr += '<td><input data-f="label" value="' + String(r.display_label||'').replace(/"/g,'&quot;') + '"/></td>';
      tr += '<td><input type="checkbox" data-f="en" ' + (r.enabled ? 'checked' : '') + '/></td>';
      tr += '<td><select data-f="model">' + modelOptions(r.model_id) + '</select></td>';
      feats.forEach(function(f) {
        var on = r.features && r.features[f.key];
        tr += '<td><input type="checkbox" data-ft="' + f.key + '" ' + (on ? 'checked' : '') + '/></td>';
      });
      tr += '<td><button type="button" data-act="save">Salvar</button> <button type="button" data-act="del">Excluir</button></td></tr>';
      return tr;
    }).join('');
    $('idTable').innerHTML = '<table><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
    $('idTable').querySelectorAll('button[data-act]').forEach(function(btn) {
      btn.onclick = async function() {
        var tr = btn.closest('tr');
        var id = tr.getAttribute('data-id');
        $('idErr').textContent = '';
        $('idErr').className = 'err';
        try {
          if (btn.getAttribute('data-act') === 'del') {
            if (!confirm('Excluir ' + id + '?')) return;
            await api('/api/identities/' + encodeURIComponent(id), { method: 'DELETE' });
          } else {
            var body = {
              display_label: tr.querySelector('input[data-f=label]').value || null,
              enabled: tr.querySelector('input[data-f=en]').checked,
              model_id: tr.querySelector('select[data-f=model]').value,
              features: {}
            };
            feats.forEach(function(f) {
              var cb = tr.querySelector('input[data-ft="' + f.key + '"]');
              body.features[f.key] = cb && cb.checked;
            });
            await api('/api/identities/' + encodeURIComponent(id), { method: 'PATCH', body: JSON.stringify(body) });
          }
          await loadIdentities();
        } catch (e) {
          $('idErr').textContent = String(e.message || e);
        }
      };
    });
  }
  async function loadTemplates() {
    var j = await api('/api/templates');
    $('tpls').innerHTML = (j.templates || []).map(function(t) {
      var safe = String(t.body).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
      return '<div style="margin:1rem 0"><strong>' + t.template_key + '</strong>' +
        '<textarea data-k="' + t.template_key + '">' + safe + '</textarea>' +
        '<button type="button" data-save="' + t.template_key + '">Salvar</button></div>';
    }).join('');
    $('tpls').querySelectorAll('button[data-save]').forEach(function(b) {
      b.onclick = async function() {
        var k = b.getAttribute('data-save');
        var ta = $('tpls').querySelector('textarea[data-k="' + k + '"]');
        await api('/api/templates/' + encodeURIComponent(k), { method: 'PUT', body: JSON.stringify({ body: ta.value }) });
        b.textContent = 'Salvo';
        setTimeout(function() { b.textContent = 'Salvar'; }, 1200);
      };
    });
  }
  $('btnAdd').onclick = async function() {
    $('idErr').textContent = '';
    $('idErr').className = 'err';
    try {
      await api('/api/identities', { method: 'POST', body: JSON.stringify({
        contact_digits: $('newId').value.trim(),
        display_label: $('newLabel').value.trim() || null,
        model_id: $('newModel').value,
        enabled: true
      })});
      $('newId').value = '';
      await loadIdentities();
    } catch (e) { $('idErr').textContent = String(e.message || e); }
  };
  $('btnReload').onclick = async function() {
    $('idErr').textContent = '';
    try {
      await api('/api/policy/reload', { method: 'POST', body: '{}' });
      $('idErr').className = 'ok';
      $('idErr').textContent = 'Cache do processo deste servidor invalidado.';
    } catch (e) {
      $('idErr').className = 'err';
      $('idErr').textContent = String(e.message || e);
    }
  };
  (async function() {
    try {
      await loadModels();
      await loadIdentities();
      await loadTemplates();
    } catch (e) {
      $('idErr').textContent = 'Falha ao carregar (token válido?). ' + (e.message || e);
    }
  })();
  </script>
</body>
</html>`;
