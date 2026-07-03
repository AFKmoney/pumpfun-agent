# GO LIVE — Premier vrai trade en 4 étapes

> Ce guide te mène du clone au premier trade réel SÉCURISÉ.
> On commence avec 0.02 SOL. Pas plus.

---

## ⚠️ AVANT DE COMMENCER

- **Wallet dédié** : crée un wallet NOUVEAU pour le bot. Pas ton wallet principal.
- **0.05-0.1 SOL** : fund ce wallet avec ce que tu peux perdre.
- **Tu es devant l'écran** : reste là pour les 30 premières minutes.

---

## Étape 1 : Préparer (5 min)

```bash
cd pumpfun-agent
git pull origin main
pip install -r requirements.txt
```

Configure ton wallet (choisis UNE méthode) :

```bash
# Méthode la plus simple : private key depuis Phantom
# Phantom → Settings → Account → Show Private Key → Copy
echo 'SOLANA_PRIVATE_KEY="5Kt...ton-clef..."' >> .env
```

Les autres secrets nécessaires dans `.env` :
```bash
HELIUS_API_KEY="hel-ton-clef-helius"
TELEGRAM_BOT_TOKEN="123:ABC..."
TELEGRAM_CHAT_ID="ton-chat-id"
```

Dans `config/config.yaml`, ajuste ces valeurs pour la sécurité :
```yaml
trading:
  mode: "live"
  fixed_size_sol: 0.02        # 0.02 SOL par trade (TINY)
  max_open_positions: 1       # une position à la fois
  max_trade_frequency_per_minute: 2

risk:
  daily_loss_cap_pct: 3.0     # stop à -3% du jour
  per_token_max_loss_pct: 20  # SL à -20% par token
```

## Étape 2 : Vérifier (1 min)

```bash
python scripts/preflight_live.py
```

Tu dois voir **8 ✅** verts. Si tu vois un ❌, fix-le avant de continuer.

```bash
python scripts/check_wallet.py
```

Confirme : bonne pubkey, solde suffisant, signature OK.

## Étape 3 : Premier trade — SINGLE SHOT (le plus sûr)

```bash
python scripts/single_shot_live.py
```

Le bot va :
1. Démarrer tous les moniteurs
2. Attendre le premier signal qui passe TOUS les gates (anti-rug + scoring + regime + trend)
3. Exécuter **UN SEUL** buy de 0.02 SOL
4. Monitorer la position (SL/TP/dev-sell/soft-rug)
5. **S'arrêter automatiquement** après que la position se ferme (ou après 30 min)

Pendant que ça tourne, observe les logs. Tu devrais voir :
- `sniping.new_launch_detected` → token trouvé
- `executor.buy` ou `executor.buy_failed` → le moment de vérité
- `latency.trace op=snipe total_ms=XX` → ta vraie latence
- Si erreur : `solana.buy.pumpfun_failed` → on apprend du vrai réseau

## Étape 4 : Si ça marche → paper prolongé

Si le single-shot fonctionne (buy atterrit, pas de crash) :
1. Remets `mode: "paper"` dans le config
2. Lance `python orchestrator.py` en paper pendant 2-4 heures
3. Observe : combien de signaux ? Combien passent les gates ? Quels exits ?
4. Mesure : winrate paper, latency p99, faux positifs

**Ensuite seulement** : remets `mode: "live"` avec `fixed_size_sol: 0.05` et laisse tourner.

---

## Dépannage du premier trade

| Symptôme | Cause probable | Fix |
|---|---|---|
| `buy_failed: custom program error: 1` | Compte ATA ou bonding curve mal résolu | Vérifie que pumpfun_ix génère les bons comptes (regarde le log `buy_ix` details) |
| `buy_failed: Blockhash not found` | RPC lent / blockhash expiré | Utilise un RPC plus rapide (Helius paid tier) |
| `buy_failed: Insufficient funds` | Pas assez de SOL pour tx + fees | Fund le wallet avec 0.05 SOL min |
| `buy_failed: Transaction simulation failed` | Format d'instruction incorrect | L'erreur contient les détails — poste-la, on debug |
| Rien ne se passe pendant 10 min | Aucun signal ne passe les gates | Normal en risk-off. Baisse `min_confidence` dans config |
| Bot détecte mais n'achète jamais | Anti-rug bloque tout (holders/liquet trop bas) | Normal sur les fresh launches. C'est la sécurité qui marche |

## Kill switch immédiat

Si quelque chose tourne mal :
```bash
touch data/KILL_SWITCH
```
Le bot s'arrête dans 2 secondes.

---

## Ce qu'on cherche à valider

1. **L'instruction pump.fun atterrit** (le plus critique — jamais testé en réel)
2. **La latence réelle** (notre cible : <800ms detect→confirm)
3. **L'anti-rug ne bloque pas tout** (sinon on ne trade jamais)
4. **Le stop-loss s'exécute** (si le token chute, on sort)
5. **Pas de crash** sur 30 min

Si ces 5 points passent, le bot est viable. Le reste c'est de l'optimisation.
