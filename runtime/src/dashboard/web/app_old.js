// QUE Guild — Web Dashboard (1:1 Godot match)
// CardsPanel: 420px right | MainPanel: 330px bottom
const CARDS_W = 420;
const MAIN_H = 330;
const TILE = 32;
const CHAR_W = 24;
const CHAR_H = 40;

// EventBus (with replay cache for ready/stats so late-joining scenes work)
const EventBus = {
  _l: {},
  _cache: {},
  on(e, fn) {
    (this._l[e] = this._l[e] || []).push(fn);
    // Replay cached event immediately for late subscribers
    if (this._cache[e]) { try { fn(this._cache[e]); } catch(x) { console.error(x); } }
  },
  emit(e, d) {
    // Cache important events
    if (e === 'ready') this._cache['ready'] = d;
    (this._l[e] || []).forEach(f => { try { f(d); } catch(x) { console.error(x); } });
  }
};

// WebSocket
let ws = null;
function connectWS() {
  ws = new WebSocket('ws://' + location.host + '/ws');
  ws.onopen = () => {
    console.log('[WS] Connected');
    EventBus.emit('ws:connected');
    sendCmd({ cmd: 'report' });
    sendCmd({ cmd: 'reports' });
  };
  ws.onmessage = (e) => {
    try {
      const d = JSON.parse(e.data);
      console.log('[WS]', d.type, d);
      if (d.type) EventBus.emit(d.type, d);
    } catch (x) { console.error('[WS] parse error', x); }
  };
  ws.onclose = () => { 
    console.log('[WS] Disconnected, reconnecting...');
    EventBus.emit('ws:disconnected'); 
    setTimeout(connectWS, 2000); 
  };
}
function sendCmd(c) { 
  if (ws && ws.readyState === 1) {
    console.log('[WS] Send:', c);
    ws.send(JSON.stringify(c)); 
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// OfficeScene — tile-based world matching Godot
// ─────────────────────────────────────────────────────────────────────────────
class OfficeScene extends Phaser.Scene {
  constructor() { 
    super('OfficeScene'); 
    this.agents = {}; 
    this.worldData = null; 
    this.tileList = [];
  }
  
  preload() {
    // All known tile types
    this.tileList = [
      'asphalt_0','asphalt_1','asphalt_2','asphalt_3','asphalt_4',
      'brownstone','brownstone_window',
      'carpet_0','carpet_1','carpet_2','carpet_3','carpet_4','carpet_5','carpet_6','carpet_7',
      'conf_carpet','door','elevator','elevator_door','glass_wall',
      'grass_0','grass_1','grass_2','grass_3','grass_4','grass_5','grass_6','grass_7',
      'hardwood_0','hardwood_1','hardwood_2','hardwood_3','hardwood_4','hardwood_5','hardwood_6','hardwood_7',
      'marble','office_door','office_wall','roof'
    ];
    this.tileList.forEach(t => this.load.image('tile_' + t, 'assets/tiles/' + t + '.png'));
    
    // Decorations
    const decos = ['desk_0','desk_1','desk_2','desk_3','office_chair',
      'potted_plant_0','potted_plant_1','potted_plant_2',
      'whiteboard','coffee_machine','water_cooler','server_rack','bookshelf'];
    decos.forEach(d => this.load.image('deco_' + d, 'assets/decos/' + d + '.png'));
    
    // Icons (may not exist, that's ok)
    this.load.image('icon_zzz', 'assets/icons/zzz.png');
    this.load.image('icon_working', 'assets/icons/working.png');
    this.load.image('icon_alert', 'assets/icons/alert.png');
    
    // Character sprites
    const agentNames = ['forge','vial','ward','lore','spark','tide','oracle','sage'];
    agentNames.forEach(a => {
      for (let d = 0; d < 4; d++) {
        for (let f = 0; f < 2; f++) {
          this.load.image(a + '_d' + d + '_f' + f, 'assets/characters/' + a + '_d' + d + '_f' + f + '.png');
        }
      }
    });
    
    // World data
    this.load.json('world', 'data/world.json');
  }

  create() {
    this.worldData = this.cache.json.get('world');
    const mapW = this.worldData.map_w || 31;
    const mapH = this.worldData.map_h || 17;
    
    // Dark background
    this.cameras.main.setBackgroundColor('#0a0c10');
    
    // Draw tile world
    const tiles = this.worldData.tiles || [];
    for (let y = 0; y < tiles.length; y++) {
      const row = tiles[y];
      for (let x = 0; x < row.length; x++) {
        const cell = row[x];
        // Handle format: { "t": "carpet", "v": 4 } -> carpet_4
        // Or: { "t": "glass_wall" } -> glass_wall
        let tileKey = cell.t || 'hardwood';
        if (cell.v !== undefined) {
          tileKey = cell.t + '_' + cell.v;
        }
        const texKey = 'tile_' + tileKey;
        if (this.textures.exists(texKey)) {
          this.add.image(x * TILE + TILE/2, y * TILE + TILE/2, texKey);
        } else {
          // Fallback: draw a colored rect
          const g = this.add.graphics();
          g.fillStyle(0x1a1e28, 1);
          g.fillRect(x * TILE, y * TILE, TILE, TILE);
          g.lineStyle(1, 0x2a3040, 0.3);
          g.strokeRect(x * TILE, y * TILE, TILE, TILE);
        }
      }
    }
    
    // Draw decorations
    const decorations = this.worldData.decorations || [];
    decorations.forEach(d => {
      const texKey = 'deco_' + d.t;
      if (this.textures.exists(texKey)) {
        this.add.image(d.x * TILE + TILE/2, d.y * TILE + TILE/2, texKey).setDepth(d.y * TILE + 50);
      }
    });
    
    // Camera fit - zoom to fill usable space
    const worldW = mapW * TILE;
    const worldH = mapH * TILE;
    const usableW = this.scale.width - CARDS_W;
    const usableH = this.scale.height - MAIN_H;
    const zoom = Math.min(usableW / worldW, usableH / worldH) * 0.95;
    this.cameras.main.setZoom(Math.max(zoom, 1));
    this.cameras.main.centerOn(worldW/2, worldH/2);
    
    // Events
    EventBus.on('ready', d => this._onReady(d));
    EventBus.on('stats', d => this._onStats(d));
    EventBus.on('walk_to', d => this._onWalkTo(d));
    EventBus.on('quest_start', d => this._onQuestStart(d));
    EventBus.on('quest_done', d => this._onQuestDone(d));
    EventBus.on('idle_action', d => this._onIdleAction(d));
    
    this.scene.launch('HUDScene');
    
    // Request fresh state from server (scenes are now ready to receive)
    setTimeout(() => sendCmd({ cmd: 'report' }), 500);
  }

  _onReady(data) {
    console.log('[OfficeScene] Ready event, spawning agents');
    const profiles = data.profiles || {};
    const homes = this.worldData.agent_homes || {};
    Object.keys(profiles).forEach(name => {
      const prof = profiles[name];
      const home = homes[name] || { x: 15, y: 8 };
      this._spawnAgent(name, prof, home);
    });
  }

  _spawnAgent(name, profile, home) {
    const lname = name.toLowerCase();
    const px = home.x * TILE + TILE/2;
    const py = home.y * TILE + TILE/2;
    
    console.log('[OfficeScene] Spawning', name, 'at', px, py);
    
    // Container
    const cont = this.add.container(px, py);
    cont.setDepth(py + 100);
    
    // Shadow
    const shadow = this.add.ellipse(0, 16, 16, 6, 0x000000, 0.4);
    cont.add(shadow);
    
    // Sprite
    const sprKey = lname + '_d0_f0';
    let sprite;
    if (this.textures.exists(sprKey)) {
      sprite = this.add.image(0, 0, sprKey).setOrigin(0.5, 0.8);
    } else {
      // Procedural fallback
      const gfx = this.make.graphics({ add: false });
      const col = Phaser.Display.Color.HexStringToColor(profile.color || '#888').color;
      gfx.fillStyle(col, 1);
      gfx.fillRoundedRect(0, 0, CHAR_W, CHAR_H, 4);
      gfx.fillStyle(0xf0d8c0, 1);
      gfx.fillCircle(CHAR_W/2, 8, 7);
      gfx.generateTexture('proc_' + name, CHAR_W, CHAR_H);
      gfx.destroy();
      sprite = this.add.image(0, 0, 'proc_' + name).setOrigin(0.5, 1);
    }
    cont.add(sprite);
    
    // Name label
    const label = this.add.text(0, 20, name, {
      fontSize: '9px', fontFamily: 'Consolas, monospace', 
      color: profile.color || '#fff',
      stroke: '#000', strokeThickness: 2
    }).setOrigin(0.5);
    cont.add(label);
    
    // State icon (optional)
    let icon = null;
    if (this.textures.exists('icon_zzz')) {
      icon = this.add.image(12, -20, 'icon_zzz').setScale(0.4);
      cont.add(icon);
    }
    
    // Bubble
    const bubble = this.add.text(0, -36, '', {
      fontSize: '9px', fontFamily: 'Consolas', color: '#e0e0e0',
      backgroundColor: '#1a1e28EE', padding: { x: 4, y: 2 },
      wordWrap: { width: 100 }
    }).setOrigin(0.5, 1).setVisible(false);
    cont.add(bubble);
    
    this.agents[name] = {
      container: cont, sprite, icon, label, bubble,
      x: px, y: py, targetX: px, targetY: py,
      dir: 0, frame: 0, frameTime: 0, state: 'idle',
      lname, bubbleTimer: 0, time: 0
    };
    
    // Animation loop
    this.time.addEvent({
      delay: 50, loop: true,
      callback: () => this._updateAgent(name, 0.05)
    });
  }

  _updateAgent(name, dt) {
    const ag = this.agents[name];
    if (!ag) return;
    
    ag.time += dt;
    
    // Movement
    const dx = ag.targetX - ag.x;
    const dy = ag.targetY - ag.y;
    const dist = Math.sqrt(dx*dx + dy*dy);
    
    if (dist > 2) {
      const speed = 60 * dt;
      const step = Math.min(speed, dist);
      ag.x += (dx / dist) * step;
      ag.y += (dy / dist) * step;
      ag.container.x = ag.x;
      ag.container.y = ag.y;
      ag.container.setDepth(ag.y + 100);
      
      // Direction (0=down, 1=left, 2=right, 3=up)
      if (Math.abs(dx) > Math.abs(dy)) {
        ag.dir = dx > 0 ? 2 : 1;
      } else {
        ag.dir = dy > 0 ? 0 : 3;
      }
      
      // Frame animation
      ag.frameTime += dt;
      if (ag.frameTime > 0.2) {
        ag.frame = 1 - ag.frame;
        ag.frameTime = 0;
      }
      
      // Walk bob
      ag.sprite.y = Math.sin(ag.time * 10) * 2;
    } else {
      // Idle sway
      ag.sprite.y = Math.sin(ag.time * 1.5) * 1;
    }
    
    // Update sprite texture
    const texKey = ag.lname + '_d' + ag.dir + '_f' + ag.frame;
    if (this.textures.exists(texKey)) ag.sprite.setTexture(texKey);
    
    // Bubble timer
    if (ag.bubbleTimer > 0) {
      ag.bubbleTimer -= dt;
      if (ag.bubbleTimer <= 0) ag.bubble.setVisible(false);
    }
  }

  _onStats(data) {
    const ag = this.agents[data.name];
    if (!ag) return;
    ag.state = data.state || 'idle';
    if (ag.icon) {
      const iconKey = ag.state === 'working' ? 'icon_working' : ag.state === 'cooldown' ? 'icon_alert' : 'icon_zzz';
      if (this.textures.exists(iconKey)) ag.icon.setTexture(iconKey);
    }
  }

  _onWalkTo(data) {
    const ag = this.agents[data.name];
    if (!ag) return;
    // data.x/y are 0-10 normalized, convert to tile coords
    const mapW = this.worldData.map_w || 31;
    const mapH = this.worldData.map_h || 17;
    ag.targetX = (data.x / 10) * mapW * TILE;
    ag.targetY = (data.y / 10) * mapH * TILE;
  }

  _onQuestStart(data) {
    const ag = this.agents[data.name];
    if (!ag) return;
    ag.state = 'working';
    this._showBubble(data.name, data.quest || 'Working...', 3);
  }

  _onQuestDone(data) {
    const ag = this.agents[data.name];
    if (!ag) return;
    ag.state = 'idle';
    this._showBubble(data.name, '✓ Done!', 2);
  }

  _onIdleAction(data) {
    const ag = this.agents[data.name];
    if (!ag) return;
    if (data.result) this._showBubble(data.name, data.result.substring(0, 40), 3);
  }

  _showBubble(name, text, duration) {
    const ag = this.agents[name];
    if (!ag) return;
    ag.bubble.setText(text);
    ag.bubble.setVisible(true);
    ag.bubbleTimer = duration;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HUDScene — exact Godot layout: 420px right, 330px bottom, 4 tabs
// ─────────────────────────────────────────────────────────────────────────────
class HUDScene extends Phaser.Scene {
  constructor() {
    super('HUDScene');
    this.cards = {};
    this.currentTab = 0;
    this.tabContent = { comms: [], quests: [], reports: [], log: [] };
    this.autoOn = false;
  }

  create() {
    const W = this.scale.width;
    const H = this.scale.height;
    
    // ── CardsPanel (right 420px) ──
    const cardsX = W - CARDS_W;
    this.add.rectangle(cardsX + CARDS_W/2, H/2, CARDS_W, H, 0x0a0c10, 0.96);
    this.add.rectangle(cardsX, H/2, 2, H, 0x2a3040, 1); // left border
    this.add.text(cardsX + CARDS_W/2, 18, 'QUE GUILD — AGENTS', {
      fontSize: '16px', fontFamily: 'Consolas, monospace', color: '#ffd700', fontStyle: 'bold'
    }).setOrigin(0.5);
    this.cardsY = 46;
    this.cardsX = cardsX + 8;
    this.cardsW = CARDS_W - 16;
    
    // ── MainPanel (bottom 330px) ──
    const mainW = W - CARDS_W;
    const mainY = H - MAIN_H;
    this.add.rectangle(mainW/2, mainY + MAIN_H/2, mainW, MAIN_H, 0x0a0c10, 0.96);
    this.add.rectangle(mainW/2, mainY, mainW, 2, 0x2a3040, 1); // top border
    
    // Buttons row
    const btnY = mainY + 12;
    const btnW = (mainW - 60) / 4;
    const btnH = 32;
    this._btn(12, btnY, btnW, btnH, 'PATROL', 0x2060cc, () => { sendCmd({cmd:'patrol'}); this._addComms('SYS','PATROL dispatched...'); });
    this._btn(12 + btnW + 8, btnY, btnW, btnH, 'AUTO', 0x208040, () => { this.autoOn = !this.autoOn; sendCmd({cmd:'auto',on:this.autoOn}); this._addComms('SYS',this.autoOn?'AUTO ON':'AUTO OFF'); });
    this._btn(12 + (btnW + 8)*2, btnY, btnW, btnH, 'REPORT', 0xcc8020, () => { sendCmd({cmd:'report'}); sendCmd({cmd:'reports'}); });
    this._btn(12 + (btnW + 8)*3, btnY, btnW, btnH, 'SAGE SCAN', 0x8020cc, () => { sendCmd({cmd:'sage_scan'}); this._addComms('SYS','SAGE deep scan...'); });
    
    // Tabs
    const tabY = mainY + 54;
    const tabW = (mainW - 20) / 4;
    this.tabs = [];
    this.tabBgs = [];
    ['COMMS','QUESTS','REPORTS','LOG'].forEach((name, i) => {
      const tx = 10 + i * tabW + tabW/2;
      const bg = this.add.rectangle(tx, tabY + 12, tabW - 4, 24, i === 0 ? 0x2a3a4a : 0x14181e, 0.95)
        .setInteractive({ useHandCursor: true });
      const txt = this.add.text(tx, tabY + 12, name, {
        fontSize: '12px', fontFamily: 'Consolas, monospace', color: i === 0 ? '#fff' : '#556677'
      }).setOrigin(0.5);
      bg.on('pointerdown', () => this._switchTab(i));
      this.tabs.push(txt);
      this.tabBgs.push(bg);
    });
    
    // Content area
    this.contentX = 12;
    this.contentY = tabY + 34;
    this.contentW = mainW - 24;
    this.contentH = MAIN_H - 130;
    
    // Tab text displays
    this.commsText = this.add.text(this.contentX, this.contentY, '[SYS] QUE Guild online.\n', {
      fontSize: '13px', fontFamily: 'Consolas, monospace', color: '#7799aa',
      wordWrap: { width: this.contentW }, lineSpacing: 2
    });
    this.questsText = this.add.text(this.contentX, this.contentY, '[SYS] Waiting for patrol...\n', {
      fontSize: '13px', fontFamily: 'Consolas, monospace', color: '#7799aa',
      wordWrap: { width: this.contentW }, lineSpacing: 2
    }).setVisible(false);
    this.reportsText = this.add.text(this.contentX, this.contentY, '[SYS] No reports yet.\n', {
      fontSize: '13px', fontFamily: 'Consolas, monospace', color: '#7799aa',
      wordWrap: { width: this.contentW }, lineSpacing: 2
    }).setVisible(false);
    this.logText = this.add.text(this.contentX, this.contentY, '[SYS] Event log ready.\n', {
      fontSize: '12px', fontFamily: 'Consolas, monospace', color: '#889988',
      wordWrap: { width: this.contentW }, lineSpacing: 2
    }).setVisible(false);
    
    // Chat input (HTML element)
    this.chatInput = document.createElement('input');
    this.chatInput.type = 'text';
    this.chatInput.placeholder = 'Message ORACLE...';
    this.chatInput.style.cssText = 'position:fixed;left:12px;bottom:12px;width:' + (mainW - 110) + 'px;height:34px;' +
      'background:#10141aF8;border:1px solid #334466;border-radius:4px;padding:0 12px;' +
      'color:#e0e8f0;font-family:Consolas,monospace;font-size:14px;outline:none;z-index:100;';
    document.body.appendChild(this.chatInput);
    
    this.chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && this.chatInput.value.trim()) {
        const text = this.chatInput.value.trim();
        this.chatInput.value = '';
        this._addComms('YOU', text);
        this._addComms('ORACLE', '...');
        sendCmd({ cmd: 'chat', text });
      }
    });
    
    // Send button
    this.add.rectangle(mainW - 50, H - 29, 80, 34, 0x2060cc, 1)
      .setInteractive({ useHandCursor: true })
      .on('pointerdown', () => {
        if (this.chatInput.value.trim()) {
          const text = this.chatInput.value.trim();
          this.chatInput.value = '';
          this._addComms('YOU', text);
          sendCmd({ cmd: 'chat', text });
        }
      });
    this.add.text(mainW - 50, H - 29, 'SEND', { fontSize: '13px', fontFamily: 'Consolas', color: '#fff' }).setOrigin(0.5);
    
    // Events
    EventBus.on('ready', d => this._onReady(d));
    EventBus.on('stats', d => this._onStats(d));
    EventBus.on('quest_start', d => this._onQuestStart(d));
    EventBus.on('quest_done', d => this._onQuestDone(d));
    EventBus.on('quest_fail', d => this._onQuestFail(d));
    EventBus.on('system', d => this._onSystem(d));
    EventBus.on('chat_response', d => this._onChatResponse(d));
    EventBus.on('log', d => this._onLog(d));
    EventBus.on('idle_action', d => this._onIdleAction(d));
  }

  _btn(x, y, w, h, label, color, onClick) {
    const bg = this.add.rectangle(x + w/2, y + h/2, w, h, color, 0.9)
      .setStrokeStyle(1, 0x3a4050)
      .setInteractive({ useHandCursor: true });
    const txt = this.add.text(x + w/2, y + h/2, label, { 
      fontSize: '12px', fontFamily: 'Consolas, monospace', color: '#ddeeff' 
    }).setOrigin(0.5);
    bg.on('pointerover', () => { bg.setFillStyle(color, 1); txt.setColor('#fff'); });
    bg.on('pointerout', () => { bg.setFillStyle(color, 0.9); txt.setColor('#ddeeff'); });
    bg.on('pointerdown', onClick);
  }

  _switchTab(idx) {
    this.currentTab = idx;
    this.tabs.forEach((t, i) => t.setColor(i === idx ? '#fff' : '#556677'));
    this.tabBgs.forEach((b, i) => b.setFillStyle(i === idx ? 0x2a3a4a : 0x14181e, 0.95));
    this.commsText.setVisible(idx === 0);
    this.questsText.setVisible(idx === 1);
    this.reportsText.setVisible(idx === 2);
    this.logText.setVisible(idx === 3);
  }

  _ts() {
    const d = new Date();
    return d.getHours().toString().padStart(2,'0') + ':' + d.getMinutes().toString().padStart(2,'0');
  }

  _addComms(who, msg) {
    this.tabContent.comms.push('[' + this._ts() + '] ' + who + ' > ' + msg);
    if (this.tabContent.comms.length > 60) this.tabContent.comms.shift();
    this.commsText.setText(this.tabContent.comms.join('\n'));
  }

  _addQuest(who, msg) {
    this.tabContent.quests.push('[' + this._ts() + '] ' + who + ': ' + msg);
    if (this.tabContent.quests.length > 60) this.tabContent.quests.shift();
    this.questsText.setText(this.tabContent.quests.join('\n'));
  }

  _addLog(who, msg) {
    this.tabContent.log.push('[' + this._ts() + '] ' + who + ': ' + msg);
    if (this.tabContent.log.length > 60) this.tabContent.log.shift();
    this.logText.setText(this.tabContent.log.join('\n'));
  }

  _onReady(data) {
    console.log('[HUDScene] Ready event, creating cards');
    const profiles = data.profiles || {};
    let yOff = 0;
    Object.keys(profiles).forEach(name => {
      this._createCard(name, profiles[name], yOff);
      yOff += 82;
    });
    this._addComms('SYS', Object.keys(profiles).length + ' agents online.');
  }

  _createCard(name, profile, yOff) {
    const x = this.cardsX;
    const y = this.cardsY + yOff;
    const w = this.cardsW;
    const h = 78;
    const colHex = profile.color || '#888888';
    const col = Phaser.Display.Color.HexStringToColor(colHex).color;
    
    // Card background
    const gfx = this.add.graphics();
    gfx.fillStyle(0x10141a, 0.95);
    gfx.fillRoundedRect(x, y, w, h, 4);
    gfx.lineStyle(3, col, 1);
    gfx.lineBetween(x + 2, y + 4, x + 2, y + h - 4);
    
    // Name + class
    this.add.text(x + 12, y + 8, name + '  ' + (profile.rpg_class || ''), {
      fontSize: '15px', fontFamily: 'Consolas, monospace', color: colHex, fontStyle: 'bold'
    });
    
    // Level
    const lvlTxt = this.add.text(x + w - 10, y + 8, 'Lv.1', {
      fontSize: '13px', fontFamily: 'Consolas, monospace', color: '#b0d090'
    }).setOrigin(1, 0);
    
    // State
    const stateTxt = this.add.text(x + w - 10, y + 24, 'IDLE', {
      fontSize: '11px', fontFamily: 'Consolas, monospace', color: '#ddaa33'
    }).setOrigin(1, 0);
    
    // Quest text
    const questTxt = this.add.text(x + 12, y + 28, '', {
      fontSize: '10px', fontFamily: 'Consolas, monospace', color: '#556677',
      wordWrap: { width: w - 80 }
    });
    
    // HP bar
    this.add.text(x + 12, y + 44, 'HP', { fontSize: '9px', fontFamily: 'Consolas', color: '#44bb55' });
    this.add.rectangle(x + 32 + (w - 52)/2, y + 48, w - 52, 8, 0x1a1a1a, 1);
    const hpBar = this.add.rectangle(x + 32, y + 48, w - 52, 8, 0x44bb55, 1).setOrigin(0, 0.5);
    
    // XP bar
    this.add.text(x + 12, y + 56, 'XP', { fontSize: '9px', fontFamily: 'Consolas', color: '#5588cc' });
    this.add.rectangle(x + 32 + (w - 52)/2, y + 60, w - 52, 8, 0x1a1a1a, 1);
    const xpBar = this.add.rectangle(x + 32, y + 60, 0, 8, 0x5588cc, 1).setOrigin(0, 0.5);
    
    // Quests count
    const questsLbl = this.add.text(x + 12, y + 68, 'Quests: 0', {
      fontSize: '9px', fontFamily: 'Consolas', color: '#556677'
    });
    
    this.cards[name] = { lvlTxt, stateTxt, questTxt, hpBar, xpBar, questsLbl, barW: w - 52 };
  }

  _onStats(data) {
    const c = this.cards[data.name];
    if (!c) return;
    c.lvlTxt.setText('Lv.' + (data.level || 1));
    const state = (data.state || 'idle').toUpperCase();
    c.stateTxt.setText(state);
    c.stateTxt.setColor(state === 'WORKING' ? '#44dd88' : state === 'COOLDOWN' ? '#8888aa' : '#ddaa33');
    c.hpBar.setSize(c.barW * Math.min((data.hp || 100) / 100, 1), 8);
    const xpMax = Math.max(data.xp_to_next || 100, 1);
    c.xpBar.setSize(c.barW * Math.min((data.xp || 0) / xpMax, 1), 8);
    c.questsLbl.setText('Quests: ' + (data.quests_completed || 0));
  }

  _onQuestStart(data) {
    const c = this.cards[data.name];
    if (c) {
      c.stateTxt.setText('WORKING').setColor('#44dd88');
      c.questTxt.setText((data.quest || '').substring(0, 35));
    }
    this._addQuest(data.name, 'started: ' + (data.quest || '').substring(0, 35));
  }

  _onQuestDone(data) {
    const c = this.cards[data.name];
    if (c) {
      c.stateTxt.setText('IDLE').setColor('#ddaa33');
      c.questTxt.setText('');
    }
    this._addComms(data.name, '✓ ' + (data.quest || '').substring(0, 30) + ' +' + (data.xp || 5) + 'XP');
    this._addQuest(data.name, '✓ done: ' + (data.quest || '').substring(0, 30));
  }

  _onQuestFail(data) {
    const c = this.cards[data.name];
    if (c) { c.stateTxt.setText('IDLE').setColor('#ddaa33'); c.questTxt.setText(''); }
    this._addQuest(data.name, '✗ failed: ' + (data.quest || '').substring(0, 30));
  }

  _onSystem(data) {
    this._addComms('SYS', data.msg || '');
  }

  _onChatResponse(data) {
    // Remove "..." placeholder
    const lines = this.tabContent.comms;
    if (lines.length && lines[lines.length-1].includes('...')) lines.pop();
    this._addComms(data.from || 'ORACLE', (data.msg || data.text || '').substring(0, 150));
  }

  _onLog(data) { this._addLog(data.name || 'SYS', data.msg || ''); }
  _onIdleAction(data) { this._addLog(data.name, 'idle: ' + (data.result || '').substring(0, 35)); }
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  const config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: window.innerWidth,
    height: window.innerHeight,
    backgroundColor: '#0a0c10',
    scene: [OfficeScene, HUDScene],
    scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH },
    render: { pixelArt: true, antialias: false }
  };
  window.game = new Phaser.Game(config);
  connectWS();
});
