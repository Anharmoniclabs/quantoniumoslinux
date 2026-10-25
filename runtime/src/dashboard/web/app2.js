// QUE Guild — Web Dashboard (1:1 Godot match)
// Phaser = map only. HUD = pure HTML overlay.
const TILE = 32;

// ── EventBus (replay-cache for late subscribers) ──
const EventBus = {
  _l: {}, _cache: {},
  on(e, fn) {
    (this._l[e] = this._l[e] || []).push(fn);
    if (this._cache[e]) { try { fn(this._cache[e]); } catch(x) { console.error(x); } }
  },
  emit(e, d) {
    if (e === 'ready') this._cache.ready = d;
    (this._l[e] || []).forEach(f => { try { f(d); } catch(x) { console.error(x); } });
  }
};

// ── WebSocket ──
let ws = null;
function connectWS() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => {
    console.log('[WS] Connected');
    sendCmd({ cmd: 'report' });
    sendCmd({ cmd: 'reports' });
  };
  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      if (d.type) EventBus.emit(d.type, d);
    } catch (x) { console.error('[WS] parse', x); }
  };
  ws.onclose = () => setTimeout(connectWS, 2000);
}
function sendCmd(c) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify(c));
}

// ── MapScene — Phaser, map only ──
class MapScene extends Phaser.Scene {
  constructor() { super('MapScene'); this.agents = {}; this.worldData = null; }

  preload() {
    // Tiles
    const tiles = [
      'asphalt_0','asphalt_1','asphalt_2','asphalt_3','asphalt_4',
      'brownstone','brownstone_window',
      'carpet_0','carpet_1','carpet_2','carpet_3','carpet_4','carpet_5','carpet_6','carpet_7',
      'conf_carpet','door','elevator','elevator_door','glass_wall',
      'grass_0','grass_1','grass_2','grass_3','grass_4','grass_5','grass_6','grass_7',
      'hardwood_0','hardwood_1','hardwood_2','hardwood_3','hardwood_4','hardwood_5','hardwood_6','hardwood_7',
      'marble','office_door','office_wall','roof'
    ];
    tiles.forEach(t => this.load.image('tile_' + t, 'assets/tiles/' + t + '.png'));

    // Decos
    ['desk_0','desk_1','desk_2','desk_3','office_chair',
     'potted_plant_0','potted_plant_1','potted_plant_2',
     'whiteboard','coffee_machine','water_cooler','server_rack','bookshelf'
    ].forEach(d => this.load.image('deco_' + d, 'assets/decos/' + d + '.png'));

    // Icons
    ['zzz','working','alert'].forEach(i => this.load.image('icon_' + i, 'assets/icons/' + i + '.png'));

    // Characters (8 agents × 4 dirs × 2 frames)
    ['forge','vial','ward','lore','spark','tide','oracle','sage'].forEach(a => {
      for (let d = 0; d < 4; d++)
        for (let f = 0; f < 2; f++)
          this.load.image(`${a}_d${d}_f${f}`, `assets/characters/${a}_d${d}_f${f}.png`);
    });

    this.load.json('world', 'data/world.json');
  }

  create() {
    this.worldData = this.cache.json.get('world');
    const W = this.worldData.map_w, H = this.worldData.map_h;
    this.cameras.main.setBackgroundColor('#0a0c10');

    // ── Tiles ──
    const tiles = this.worldData.tiles || [];
    for (let y = 0; y < tiles.length; y++) {
      for (let x = 0; x < tiles[y].length; x++) {
        const c = tiles[y][x];
        let k = c.t || 'hardwood';
        if (c.v !== undefined) k = c.t + '_' + c.v;
        const tex = 'tile_' + k;
        if (this.textures.exists(tex)) {
          this.add.image(x * TILE + TILE/2, y * TILE + TILE/2, tex);
        } else {
          const g = this.add.graphics();
          g.fillStyle(0x1a1e28, 1); g.fillRect(x*TILE, y*TILE, TILE, TILE);
          g.lineStyle(1, 0x2a3040, 0.3); g.strokeRect(x*TILE, y*TILE, TILE, TILE);
        }
      }
    }

    // ── Decorations ──
    (this.worldData.decorations || []).forEach(d => {
      // world.json uses {type, variant} not {t}
      const key = 'deco_' + d.type + (d.variant ? '_' + d.variant : '');
      const keySimple = 'deco_' + d.type;
      const tex = this.textures.exists(key) ? key : this.textures.exists(keySimple) ? keySimple : null;
      if (tex) this.add.image(d.x * TILE + TILE/2, d.y * TILE + TILE/2, tex).setDepth(d.y * TILE + 50);
    });

    // ── Room labels ──
    (this.worldData.buildings || []).forEach(b => {
      const cx = (b.x + b.w / 2) * TILE;
      const cy = b.y * TILE - 4;
      this.add.text(cx, cy, b.label, {
        fontSize: '10px', fontFamily: 'Consolas, monospace', color: '#8899aa',
        stroke: '#000', strokeThickness: 3
      }).setOrigin(0.5, 1).setDepth(9999);
    });

    // ── Camera: fill the map canvas area, centered ──
    const worldPx = W * TILE, worldPy = H * TILE;
    const canvasW = this.scale.width, canvasH = this.scale.height;
    const zoom = Math.min(canvasW / worldPx, canvasH / worldPy);
    this.cameras.main.setZoom(zoom);
    this.cameras.main.centerOn(worldPx / 2, worldPy / 2);

    // ── Events ──
    EventBus.on('ready', d => this._onReady(d));
    EventBus.on('stats', d => this._onStats(d));
    EventBus.on('walk_to', d => this._onWalkTo(d));
    EventBus.on('quest_start', d => { const a = this.agents[d.name]; if (a) { a.state = 'working'; this._bubble(d.name, d.quest || 'Working...', 3); } });
    EventBus.on('quest_done', d => { const a = this.agents[d.name]; if (a) { a.state = 'idle'; this._bubble(d.name, '✓ Done!', 2); } });
    EventBus.on('idle_action', d => { if (d.result) this._bubble(d.name, d.result.substring(0, 40), 3); });

    // Delayed report request (after HUD registers too)
    setTimeout(() => sendCmd({ cmd: 'report' }), 800);
  }

  _onReady(data) {
    const profiles = data.profiles || {};
    const homes = this.worldData.agent_homes || {};
    Object.keys(profiles).forEach(name => {
      if (this.agents[name]) return; // don't double-spawn
      this._spawn(name, profiles[name], homes[name] || { x: 15, y: 8 });
    });
  }

  _spawn(name, profile, home) {
    const ln = name.toLowerCase();
    const px = home.x * TILE + TILE/2, py = home.y * TILE + TILE/2;
    const cont = this.add.container(px, py).setDepth(py + 100);

    // Shadow
    cont.add(this.add.ellipse(0, 16, 16, 6, 0x000000, 0.4));

    // Sprite
    const sKey = ln + '_d0_f0';
    let sprite;
    if (this.textures.exists(sKey)) {
      sprite = this.add.image(0, 0, sKey).setOrigin(0.5, 0.8);
    } else {
      const g = this.make.graphics({ add: false });
      const col = Phaser.Display.Color.HexStringToColor(profile.color || '#888').color;
      g.fillStyle(col, 1); g.fillRoundedRect(0, 0, 24, 40, 4);
      g.fillStyle(0xf0d8c0, 1); g.fillCircle(12, 8, 7);
      g.generateTexture('proc_' + name, 24, 40); g.destroy();
      sprite = this.add.image(0, 0, 'proc_' + name).setOrigin(0.5, 1);
    }
    cont.add(sprite);

    // Name
    cont.add(this.add.text(0, 20, name, {
      fontSize: '9px', fontFamily: 'Consolas, monospace', color: profile.color || '#fff',
      stroke: '#000', strokeThickness: 2
    }).setOrigin(0.5));

    // Icon
    let icon = null;
    if (this.textures.exists('icon_zzz')) {
      icon = this.add.image(12, -20, 'icon_zzz').setScale(0.4);
      cont.add(icon);
    }

    // Bubble
    const bubble = this.add.text(0, -36, '', {
      fontSize: '9px', fontFamily: 'Consolas', color: '#e0e0e0',
      backgroundColor: '#1a1e28EE', padding: { x: 4, y: 2 }, wordWrap: { width: 100 }
    }).setOrigin(0.5, 1).setVisible(false);
    cont.add(bubble);

    this.agents[name] = {
      container: cont, sprite, icon, bubble,
      x: px, y: py, tx: px, ty: py,
      dir: 0, frame: 0, ft: 0, state: 'idle', ln, bt: 0, t: 0
    };

    this.time.addEvent({ delay: 50, loop: true, callback: () => this._tick(name, 0.05) });
  }

  _tick(name, dt) {
    const a = this.agents[name]; if (!a) return;
    a.t += dt;
    const dx = a.tx - a.x, dy = a.ty - a.y, dist = Math.sqrt(dx*dx + dy*dy);
    if (dist > 2) {
      const step = Math.min(60 * dt, dist);
      a.x += (dx/dist)*step; a.y += (dy/dist)*step;
      a.container.setPosition(a.x, a.y).setDepth(a.y + 100);
      a.dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 2 : 1) : (dy > 0 ? 0 : 3);
      a.ft += dt; if (a.ft > 0.2) { a.frame = 1 - a.frame; a.ft = 0; }
      a.sprite.y = Math.sin(a.t * 10) * 2;
    } else {
      a.sprite.y = Math.sin(a.t * 1.5) * 1;
    }
    const tk = `${a.ln}_d${a.dir}_f${a.frame}`;
    if (this.textures.exists(tk)) a.sprite.setTexture(tk);
    if (a.bt > 0) { a.bt -= dt; if (a.bt <= 0) a.bubble.setVisible(false); }
  }

  _onStats(data) {
    const a = this.agents[data.name]; if (!a) return;
    a.state = data.state || 'idle';
    if (a.icon) {
      const k = a.state === 'working' ? 'icon_working' : a.state === 'cooldown' ? 'icon_alert' : 'icon_zzz';
      if (this.textures.exists(k)) a.icon.setTexture(k);
    }
  }

  _onWalkTo(data) {
    const a = this.agents[data.name]; if (!a) return;
    const W = this.worldData.map_w, H = this.worldData.map_h;
    a.tx = (data.x / 10) * W * TILE; a.ty = (data.y / 10) * H * TILE;
  }

  _bubble(name, text, dur) {
    const a = this.agents[name]; if (!a) return;
    a.bubble.setText(text).setVisible(true); a.bt = dur;
  }
}

// ── HUD (pure HTML) ──
function buildHUD() {
  const CARDS_W = 300;
  const MAIN_H = 220;

  // Right panel
  const rightPanel = document.createElement('div');
  rightPanel.id = 'right-panel';
  rightPanel.innerHTML = '<div class="panel-title">QUE GUILD — AGENTS</div><div id="agent-cards"></div>';
  document.body.appendChild(rightPanel);

  // Bottom panel
  const bottomPanel = document.createElement('div');
  bottomPanel.id = 'bottom-panel';
  bottomPanel.innerHTML = `
    <div class="btn-row">
      <button class="hud-btn btn-blue" onclick="sendCmd({cmd:'patrol'});addComms('SYS','PATROL dispatched...')">PATROL</button>
      <button class="hud-btn btn-green" id="btn-auto">AUTO</button>
      <button class="hud-btn btn-orange" onclick="sendCmd({cmd:'report'});sendCmd({cmd:'reports'})">REPORT</button>
      <button class="hud-btn btn-purple" onclick="sendCmd({cmd:'sage_scan'});addComms('SYS','SAGE deep scan...')">SAGE SCAN</button>
    </div>
    <div class="tab-row">
      <div class="tab active" data-tab="0">COMMS</div>
      <div class="tab" data-tab="1">QUESTS</div>
      <div class="tab" data-tab="2">REPORTS</div>
      <div class="tab" data-tab="3">LOG</div>
    </div>
    <div class="tab-content" id="tab-content">
      <div class="tab-pane active" id="pane-comms">[SYS] QUE Guild online.</div>
      <div class="tab-pane" id="pane-quests"></div>
      <div class="tab-pane" id="pane-reports"></div>
      <div class="tab-pane" id="pane-log"></div>
    </div>
    <div class="chat-row">
      <input type="text" id="chat-input" placeholder="Message ORACLE..." />
      <button class="hud-btn btn-blue" id="btn-send">SEND</button>
    </div>
  `;
  document.body.appendChild(bottomPanel);

  // Tab switching
  bottomPanel.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      bottomPanel.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      bottomPanel.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('pane-' + ['comms','quests','reports','log'][tab.dataset.tab]).classList.add('active');
    });
  });

  // Auto toggle
  let autoOn = false;
  document.getElementById('btn-auto').addEventListener('click', () => {
    autoOn = !autoOn;
    sendCmd({ cmd: 'auto', on: autoOn });
    addComms('SYS', autoOn ? 'AUTO ON' : 'AUTO OFF');
    document.getElementById('btn-auto').textContent = autoOn ? 'AUTO ■' : 'AUTO';
  });

  // Chat
  const chatInput = document.getElementById('chat-input');
  const sendChat = () => {
    const text = chatInput.value.trim();
    if (!text) return;
    chatInput.value = '';
    addComms('YOU', text);
    sendCmd({ cmd: 'chat', text });
  };
  chatInput.addEventListener('keydown', e => { if (e.key === 'Enter') sendChat(); });
  document.getElementById('btn-send').addEventListener('click', sendChat);

  // ── Event handlers ──
  const cards = {};

  EventBus.on('ready', d => {
    const profiles = d.profiles || {};
    const container = document.getElementById('agent-cards');
    container.innerHTML = '';
    Object.keys(profiles).forEach(name => {
      const p = profiles[name];
      const card = document.createElement('div');
      card.className = 'agent-card';
      card.style.borderLeftColor = p.color || '#888';
      card.innerHTML = `
        <div class="card-header">
          <span class="card-name" style="color:${p.color}">${name}  <span class="card-class">${p.rpg_class || ''}</span></span>
          <span class="card-level">Lv.1</span>
          <span class="card-state">IDLE</span>
        </div>
        <div class="card-quest"></div>
        <div class="bar-row"><span class="bar-label hp-label">HP</span><div class="bar-bg"><div class="bar-fill hp-bar" style="width:100%"></div></div></div>
        <div class="bar-row"><span class="bar-label xp-label">XP</span><div class="bar-bg"><div class="bar-fill xp-bar" style="width:0%"></div></div></div>
        <div class="card-quests-count">Quests: 0</div>
      `;
      container.appendChild(card);
      cards[name] = card;
    });
    addComms('SYS', Object.keys(profiles).length + ' agents online.');
  });

  EventBus.on('stats', d => {
    const card = cards[d.name]; if (!card) return;
    card.querySelector('.card-level').textContent = 'Lv.' + (d.level || 1);
    const state = (d.state || 'idle').toUpperCase();
    const stateEl = card.querySelector('.card-state');
    stateEl.textContent = state;
    stateEl.className = 'card-state state-' + state.toLowerCase();
    card.querySelector('.hp-bar').style.width = Math.min(d.hp || 100, 100) + '%';
    const xpMax = Math.max(d.xp_to_next || 100, 1);
    card.querySelector('.xp-bar').style.width = Math.min((d.xp || 0) / xpMax * 100, 100) + '%';
    card.querySelector('.card-quests-count').textContent = 'Quests: ' + (d.quests_completed || 0);
  });

  EventBus.on('quest_start', d => {
    const card = cards[d.name]; if (card) {
      card.querySelector('.card-state').textContent = 'WORKING';
      card.querySelector('.card-state').className = 'card-state state-working';
      card.querySelector('.card-quest').textContent = (d.quest || '').substring(0, 40);
    }
    addQuest(d.name, 'started: ' + (d.quest || '').substring(0, 40));
  });

  EventBus.on('quest_done', d => {
    const card = cards[d.name]; if (card) {
      card.querySelector('.card-state').textContent = 'IDLE';
      card.querySelector('.card-state').className = 'card-state state-idle';
      card.querySelector('.card-quest').textContent = '';
    }
    addComms(d.name, '✓ ' + (d.quest || '').substring(0, 30) + ' +' + (d.xp || 5) + 'XP');
    addQuest(d.name, '✓ done: ' + (d.quest || '').substring(0, 30));
  });

  EventBus.on('quest_fail', d => {
    const card = cards[d.name]; if (card) {
      card.querySelector('.card-state').textContent = 'IDLE';
      card.querySelector('.card-state').className = 'card-state state-idle';
      card.querySelector('.card-quest').textContent = '';
    }
    addQuest(d.name, '✗ failed: ' + (d.quest || '').substring(0, 30));
  });

  EventBus.on('system', d => addComms('SYS', d.msg || ''));
  EventBus.on('chat_response', d => {
    addComms(d.from || 'ORACLE', (d.msg || d.text || '').substring(0, 200));
  });
  EventBus.on('log', d => addLog(d.name || 'SYS', d.msg || ''));
  EventBus.on('idle_action', d => addLog(d.name, 'idle: ' + (d.result || '').substring(0, 40)));
}

// ── Log helpers ──
function _ts() {
  const d = new Date();
  return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
}

function _appendPane(id, text, maxLines) {
  const el = document.getElementById(id);
  if (!el) return;
  const lines = el.textContent.split('\n');
  lines.push(text);
  while (lines.length > (maxLines || 80)) lines.shift();
  el.textContent = lines.join('\n');
  el.scrollTop = el.scrollHeight;
}

function addComms(who, msg) { _appendPane('pane-comms', `[${_ts()}] ${who} > ${msg}`); }
function addQuest(who, msg) { _appendPane('pane-quests', `[${_ts()}] ${who}: ${msg}`); }
function addLog(who, msg)   { _appendPane('pane-log', `[${_ts()}] ${who}: ${msg}`); }

// ── Boot ──
window.addEventListener('load', () => {
  buildHUD();

  const mapEl = document.getElementById('map-container');
  const config = {
    type: Phaser.AUTO,
    parent: 'map-container',
    width: mapEl.clientWidth,
    height: mapEl.clientHeight,
    backgroundColor: '#0a0c10',
    scene: [MapScene],
    scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH },
    render: { pixelArt: true, antialias: false }
  };
  window.game = new Phaser.Game(config);
  connectWS();
});
