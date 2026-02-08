---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
math: mathjax
style: |
  /* Global Styles - Zoomed out */
  section {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    font-size: 18px;
    padding: 30px 40px;
    line-height: 1.4;
  }
  
  /* Title Slide */
  section.title {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    text-align: center;
  }
  section.title h1 { font-size: 2.4em; margin-bottom: 0.2em; border: none; color: #e94560; }
  section.title h2 { font-size: 1.4em; font-weight: 600; border: none; color: #64b5f6 !important; }
  section.title p { color: #aaa; font-size: 0.85em; }
  
  /* Section Divider */
  section.divider {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }
  section.divider h1 { font-size: 2.2em; border: none; text-align: center; }
  section.divider p { font-size: 1.1em; opacity: 0.9; }
  
  /* Headings */
  h1 { color: #1a1a2e; font-size: 1.5em; border-bottom: 3px solid #e94560; padding-bottom: 8px; margin-bottom: 15px; }
  h2 { color: #16213e; font-size: 1.1em; margin-top: 12px; margin-bottom: 8px; }
  h3 { color: #0f3460; font-size: 0.95em; margin-top: 8px; }
  
  /* Layout */
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 10px; align-items: start; }
  .columns-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; align-items: start; }
  .columns-stretch { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-top: 10px; align-items: stretch; }
  .center { text-align: center; }
  
  /* Cards */
  .card { background: #f8f9fa; border-radius: 10px; padding: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); font-size: 0.9em; height: fit-content; }
  .card-blue { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 4px solid #1976d2; }
  .card-green { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-left: 4px solid #388e3c; }
  .card-orange { background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); border-left: 4px solid #f57c00; }
  .card-purple { background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); border-left: 4px solid #7b1fa2; }
  .card-red { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-left: 4px solid #d32f2f; }
  
  /* Boxes */
  .concept { background: #e0f7fa; border-left: 4px solid #00838f; padding: 12px 15px; margin: 12px 0; border-radius: 0 8px 8px 0; font-size: 0.9em; }
  .formula { background: #fafafa; border: 2px solid #e0e0e0; border-radius: 8px; padding: 12px; text-align: center; margin: 12px 0; }
  .success { background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); border-left: 4px solid #4caf50; padding: 12px; border-radius: 0 8px 8px 0; margin: 8px 0; font-size: 0.9em; }
  .warning { background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); border-left: 4px solid #ff9800; padding: 12px; border-radius: 0 8px 8px 0; font-size: 0.9em; }
  .info { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border-left: 4px solid #2196f3; padding: 12px; border-radius: 0 8px 8px 0; font-size: 0.9em; }
  
  /* Tables */
  table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.8em; }
  th { background: #37474f; color: white; padding: 8px 10px; text-align: left; }
  td { padding: 6px 10px; border-bottom: 1px solid #e0e0e0; }
  tr:nth-child(even) { background: #f5f5f5; }
  
  /* Code - Light Theme */
  code { background: #f5f5f5; color: #c7254e; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; border: 1px solid #e0e0e0; }
  pre { background: #fafafa; color: #333; padding: 12px; border-radius: 8px; border: 1px solid #e0e0e0; font-size: 0.72em; line-height: 1.4; }
  pre code { background: none; border: none; color: #333; }
  
  /* Metrics */
  .metric { text-align: center; padding: 15px; }
  .metric-value { font-size: 2em; font-weight: bold; color: #e94560; }
  .metric-label { font-size: 0.8em; color: #666; margin-top: 4px; }
  
  /* Enhanced Diagrams */
  .diagram { background: linear-gradient(180deg, #fafafa 0%, #f0f0f0 100%); border-radius: 12px; padding: 20px; margin: 10px 0; border: 1px solid #e0e0e0; }
  .diagram-title { font-weight: 600; font-size: 0.75em; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; text-align: center; }
  
  /* Schema boxes */
  .schema { display: flex; flex-direction: column; gap: 6px; }
  .schema-row { display: flex; align-items: center; justify-content: center; gap: 10px; }
  .schema-node { padding: 8px 14px; border-radius: 8px; font-size: 0.75em; font-weight: 600; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 80px; }
  .schema-node.primary { background: linear-gradient(135deg, #1976d2, #1565c0); color: white; }
  .schema-node.secondary { background: linear-gradient(135deg, #43a047, #388e3c); color: white; }
  .schema-node.accent { background: linear-gradient(135deg, #fb8c00, #f57c00); color: white; }
  .schema-node.highlight { background: linear-gradient(135deg, #8e24aa, #7b1fa2); color: white; }
  .schema-node.danger { background: linear-gradient(135deg, #e53935, #d32f2f); color: white; }
  .schema-node.neutral { background: linear-gradient(135deg, #757575, #616161); color: white; }
  .schema-node.light { background: white; border: 2px solid #bdbdbd; color: #424242; }
  .schema-node.large { padding: 12px 20px; font-size: 0.85em; min-width: 120px; }
  .schema-node.small { padding: 5px 10px; font-size: 0.7em; min-width: 60px; }
  .schema-arrow { color: #9e9e9e; font-size: 1.2em; font-weight: bold; }
  .schema-arrow-label { font-size: 0.65em; color: #757575; font-style: italic; }
  .schema-bracket { border: 2px dashed #bdbdbd; border-radius: 8px; padding: 10px; margin: 5px 0; }
  .schema-bracket-label { font-size: 0.65em; color: #757575; text-align: center; margin-top: 5px; }
  
  /* Decision tree */
  .decision { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .decision-node { padding: 6px 12px; border-radius: 6px; font-size: 0.7em; font-weight: 500; }
  .decision-diamond { background: #fff3e0; border: 2px solid #ff9800; color: #e65100; transform: rotate(0deg); padding: 8px 15px; }
  .decision-yes { color: #388e3c; font-size: 0.65em; font-weight: 600; }
  .decision-no { color: #d32f2f; font-size: 0.65em; font-weight: 600; }
  
  /* Architecture diagram */
  .arch-diagram { display: flex; flex-direction: column; gap: 8px; }
  .arch-layer { background: linear-gradient(90deg, #fafafa, #f5f5f5); border-radius: 10px; padding: 10px 15px; border: 1px solid #e0e0e0; display: flex; align-items: center; gap: 15px; }
  .arch-layer-icon { font-size: 1.3em; }
  .arch-layer-title { font-weight: 600; font-size: 0.7em; color: #424242; min-width: 90px; text-transform: uppercase; letter-spacing: 0.5px; }
  .arch-components { display: flex; gap: 8px; flex-wrap: wrap; flex: 1; }
  .arch-box { padding: 6px 10px; border-radius: 6px; font-size: 0.7em; font-weight: 500; display: flex; align-items: center; gap: 5px; }
  .arch-box.ingest { background: #fff3e0; border: 1px solid #ffb74d; color: #e65100; }
  .arch-box.process { background: #f3e5f5; border: 1px solid #ba68c8; color: #7b1fa2; }
  .arch-box.storage { background: #e8f5e9; border: 1px solid #81c784; color: #2e7d32; }
  .arch-box.serve { background: #e3f2fd; border: 1px solid #64b5f6; color: #1565c0; }
  .arch-box.client { background: #fce4ec; border: 1px solid #f48fb1; color: #c2185b; }
  .arch-connector { text-align: center; color: #9e9e9e; font-size: 0.9em; padding: 2px 0; }
  
  /* Matrix visualization */
  .matrix { display: inline-block; border: 2px solid #424242; border-radius: 4px; padding: 4px; margin: 0 5px; }
  .matrix-label { font-size: 0.7em; color: #616161; text-align: center; margin-top: 3px; }
  
  /* Pipeline */
  .pipeline { display: flex; align-items: center; justify-content: center; gap: 4px; flex-wrap: wrap; }
  .pipeline-step { display: flex; flex-direction: column; align-items: center; }
  .pipeline-num { background: #424242; color: white; width: 18px; height: 18px; border-radius: 50%; font-size: 0.65em; display: flex; align-items: center; justify-content: center; margin-bottom: 4px; }
  .pipeline-box { padding: 6px 10px; border-radius: 6px; font-size: 0.7em; font-weight: 500; text-align: center; }
  .pipeline-arrow { color: #bdbdbd; font-size: 1.1em; }
---

<!-- _class: title -->

# Système de Recommandation d'Articles

## Content-Based & Collaborative Filtering

**Projet 10 - OpenClassrooms**

---

# Agenda

<div class="columns">

<div>

### Partie 1 : Fondamentaux
1. **Contexte & Objectifs**
2. **Exploration des Données**
3. **Approches de Recommandation**

### Partie 2 : Implémentation
4. **Content-Based Filtering**
5. **Collaborative Filtering**
6. **Approche Hybride**

</div>

<div>

### Partie 3 : Production
7. **Architecture Cloud**
8. **Déploiement Azure**

### Partie 4 : Résultats
9. **Métriques & Évaluation**
10. **Conclusions & Perspectives**

</div>

</div>

---

<!-- _class: divider -->

# Contexte & Objectifs

Comprendre le besoin métier et les objectifs techniques

---

# Contexte Métier

<div class="columns">

<div>

## Le Défi

Une plateforme de contenu avec **des millions d'articles** :

<div class="warning">

**Comment aider les utilisateurs à découvrir du contenu pertinent parmi une masse d'informations ?**

</div>

### Impact Business
- Augmentation du **temps passé** sur la plateforme
- Amélioration de l'**engagement utilisateur**
- Réduction du **taux de rebond**

</div>

<div>

## La Solution

Un **système de recommandation intelligent** :

<div class="card card-blue">

- Analyse le **comportement** des utilisateurs
- Comprend le **contenu** des articles
- Propose des recommandations **personnalisées**

</div>

### Approche Choisie
Combiner **deux méthodes complémentaires** :
1. Filtrage basé sur le contenu
2. Filtrage collaboratif

</div>

</div>

---

# Objectifs du Projet

<div class="columns-3">

<div class="card card-blue">

### Fonctionnel
- Recommander **5 articles** pertinents
- Gérer le **cold-start**
- Temps de réponse **< 500ms**

</div>

<div class="card card-green">

### Technique
- **Content-Based Filtering**
- **Collaborative Filtering**
- Approche **Hybride**

</div>

<div class="card card-purple">

### Production
- **Azure Functions**
- Serverless scalable
- **Pulumi** IaC

</div>

</div>

<div class="concept">

**Critères de Succès** : RMSE < 0.20 pour le filtrage collaboratif, API fonctionnelle avec monitoring

</div>

---

<!-- _class: divider -->

# Exploration des Données

Comprendre les données disponibles

---

# Jeux de Données

<div class="columns">

<div>

## Articles

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre d'articles | **364,047** |
| Embeddings | 250 dimensions |
| Metadonnées | Titre, catégorie |

### Embeddings Pré-calculés
Vecteurs denses représentant le **contenu sémantique** :

```python
article_1 = [0.12, -0.45, 0.78, ...]
article_2 = [0.34, 0.21, -0.56, ...]
```

</div>

<div>

## Interactions Utilisateurs

| Caractéristique | Valeur |
|-----------------|--------|
| Utilisateurs | **~320,000** |
| Interactions | **~2.9 millions** |
| Type | Clics (implicite) |

### Structure des Clics
- `user_id` : Identifiant utilisateur
- `click_article_id` : Article consulté
- `session_id` : Session de navigation
- `click_timestamp` : Horodatage

</div>

</div>

---

# Distribution des Interactions

<div class="columns">

<div>

## Problématique : Données Sparse

<div class="warning">

La matrice User-Item est **extrêmement sparse** :
- Densité ≈ **0.003%**
- La plupart des utilisateurs ont < 10 clics

</div>

### Conséquences
- Difficulté pour le **filtrage collaboratif**
- Importance du **cold-start handling**
- Nécessité d'une approche **hybride**

</div>

<div>

## Distribution Long-Tail

<div class="card card-orange">

### Articles Populaires
**20%** des articles génèrent **80%** des clics

### Utilisateurs Actifs
Forte variance → Normalisation nécessaire

</div>

<div class="concept">

**Insight** : Clics → transformation en "ratings" nécessaire

</div>

</div>

</div>

---

<!-- _class: divider -->

# Approches de Recommandation

Vue d'ensemble des méthodes

---

# Panorama des Méthodes

<div class="columns">

<div class="card card-blue">
<h3 style="margin-top:0; color:#1565c0;">Content-Based Filtering</h3>

**Principe** : Recommander des articles similaires à ceux déjà consultés

**Avantages** : Pas besoin d'autres utilisateurs, gère le cold-start items

**Limites** : Manque de sérendipité, bulle de filtre
</div>

<div class="card card-green">
<h3 style="margin-top:0; color:#2e7d32;">Collaborative Filtering</h3>

**Principe** : Exploiter les comportements similaires entre utilisateurs

**Avantages** : Découvre des patterns cachés, sérendipité

**Limites** : Cold-start difficile, nécessite beaucoup de données
</div>

</div>

<div class="info" style="margin-top: 20px;">

**Notre Approche** : Combiner les deux dans un système **hybride** pour profiter de leurs avantages respectifs.

</div>

---

<!-- _class: divider -->

# Content-Based Filtering

Recommandations basées sur le contenu

---

# Principe du Content-Based

<div class="columns">

<div>

## Comment ça marche ?

1. **Représenter** chaque article par un vecteur
2. **Calculer** la similarité entre articles
3. **Recommander** les articles similaires

<div class="formula">

$$\text{sim}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$
**Similarité Cosinus**

</div>

</div>

<div>

## Pipeline Content-Based

<div class="diagram">
<div class="schema">
<div class="schema-row"><div class="schema-node primary large">👤 User lit Article A</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node light">📊 Embedding A [250 dims]</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node accent">🔄 Cosine Similarity vs 364K articles</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node highlight">📈 Ranking par score</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node secondary large">✅ Top 5 Articles Recommandés</div></div>
</div>
</div>

</div>

</div>

---

# Implémentation Content-Based

<div class="columns">

<div>

## 3 Stratégies de Profil Utilisateur

<div class="card card-orange" style="margin-bottom:8px;">

### Last Click (1-2 clics)
On prend **le dernier article lu** et on cherche les articles les plus similaires (cosine similarity). Simple et direct quand on a peu de signal.

</div>

<div class="card card-blue" style="margin-bottom:8px;">

### Combined (3-10 clics)
Pour chacun des **3 derniers articles** lus, on récupère ses 3 articles les plus similaires. On fusionne ces 9 candidats et on garde les meilleurs scores. Cela couvre **plusieurs centres d'intérêt** de l'utilisateur.

</div>

<div class="card card-green">

### Average (>10 clics)
On calcule la **moyenne des embeddings** des 10 derniers articles → un seul vecteur "profil utilisateur". On cherche les articles les plus proches de ce profil global.

</div>

</div>

<div>

## Gestion du Cold-Start

<div class="card card-orange">

### Nouvel Utilisateur (0 clic)
**Fallback** : articles les plus **populaires**

</div>

<div class="card card-blue" style="margin-top: 10px;">

### Utilisateur avec ≥ 1 clic
**CB immédiat** : le dernier clic suffit pour trouver des articles similaires via cosine similarity

</div>

<div class="card card-green" style="margin-top: 10px;">

### Nouvel Article
**Géré nativement** : son embedding est disponible dès la publication, recommandable immédiatement

</div>

</div>

</div>

---

<!-- _class: divider -->

# Collaborative Filtering

Exploiter l'intelligence collective

---

# Principe du Collaborative Filtering

<div class="columns">

<div>

## L'Intuition

<div class="concept">

**"Les utilisateurs qui ont aimé les mêmes articles aiméront probablement les mêmes articles à l'avenir"**

</div>

### Matrice User-Item

|  | Art.1 | Art.2 | Art.3 | Art.4 |
|--|-------|-------|-------|-------|
| User A | 0.8 | 0.6 | **?** | 0.2 |
| User B | 0.7 | 0.5 | 0.9 | **?** |
| User C | **?** | 0.4 | 0.8 | 0.3 |

**Objectif** : Prédire les `?`

</div>

<div>

## Deux Approches

### Memory-Based vs Model-Based

<div class="diagram" style="padding: 12px;">
<div class="schema">
<div class="schema-row"><div class="schema-node neutral large">📊 Matrice User-Item (Sparse 0.003%)</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row">
<div class="schema-bracket" style="display:flex; gap:15px; padding:12px;">
<div style="text-align:center;">
<div class="schema-node accent">KNN</div>
<div class="schema-bracket-label">O(n) - Lent</div>
</div>
<div style="text-align:center;">
<div class="schema-node secondary">SVD ✓</div>
<div class="schema-bracket-label">O(1) - Rapide</div>
</div>
</div>
</div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node primary">🎯 Prédictions de ratings</div></div>
</div>
</div>

</div>

</div>

---

# Transformation Clics → Ratings

<div class="columns">

<div>

## Formule de Rating Implicite

<div class="warning" style="margin-bottom: 15px;">

**Problème** : Nous n'avons que des clics, pas de notes

</div>

<div class="formula">

$$\text{rating} = \text{base} \times \text{session} \times \text{recency}$$

</div>

### Composants
- **base** : $1 + \log(1 + \text{clicks})$
- **session** : $1 + \frac{\text{n\_sessions}}{\text{clicks}}$
- **recency** : $1 + e^{-0.1 \times \text{days}}$

</div>

<div>

## Normalisation

<div class="card card-blue">

### Pourquoi normaliser ?
Éviter le **biais des utilisateurs actifs** - un utilisateur avec 1000 clics ne doit pas dominer

### Méthode
- Échelle logarithmique (dampen high activity)
- Bonus diversité sessions
- Poids récence (clicks récents > anciens)

</div>

</div>

</div>

---

# Implémentation : Vectorisation

<div class="columns">

<div>

## Problème de Performance

<div class="card card-red">

### Approche Naive
```python
df.groupby([...]).apply(func)
```
- **2.9M lignes** → itérations Python
- Temps : **10-30+ minutes** ❌

</div>

### Solution : Opérations Vectorisées

<div class="card card-green">

```python
# Agrégations vectorisées
df.groupby([...]).agg({
    'session_size': 'sum',
    'session_id': 'nunique',
    'recency': 'mean'
})
```
- Opérations C-level optimisées
- Temps : **5-15 secondes** ✅

</div>

</div>

<div>

## Pipeline Vectorisé

<div class="diagram" style="padding: 12px;">
<div class="schema">
<div class="schema-row"><div class="schema-node primary">📊 df_clicks (2.9M rows)</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node light small">Pre-calcul: recency_factor</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node accent">groupby().agg() - Sessions</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node accent">groupby().agg() - Clicks</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node light small">Merge + Calcul vectoriel</div></div>
<div class="schema-row"><span class="schema-arrow">↓</span></div>
<div class="schema-row"><div class="schema-node secondary">✅ Ratings (2.9M) en ~10s</div></div>
</div>
</div>

<div class="success" style="margin-top:10px;">

**Speedup : 100x+** grâce à la vectorisation

</div>

</div>

</div>

---

# SVD : Factorisation Matricielle

<div class="columns">

<div>

## Le Concept

<div class="formula">

$$R \approx P \times Q^T$$

</div>

- **R** : Matrice User-Item (sparse)
- **P** : Facteurs utilisateurs (n_users × k)
- **Q** : Facteurs items (n_items × k)

<div class="diagram" style="padding: 12px; margin-top: 15px;">
<div class="schema">
<div class="schema-row" style="gap: 8px; align-items: center;">
<div style="text-align:center;">
<div class="schema-node danger" style="min-width:70px;">R</div>
<div style="font-size:0.6em;color:#666;">320K × 364K<br/>Sparse</div>
</div>
<span class="schema-arrow">≈</span>
<div style="text-align:center;">
<div class="schema-node highlight" style="min-width:50px;">P</div>
<div style="font-size:0.6em;color:#666;">320K × 50</div>
</div>
<span class="schema-arrow">×</span>
<div style="text-align:center;">
<div class="schema-node highlight" style="min-width:50px;">Qᵀ</div>
<div style="font-size:0.6em;color:#666;">50 × 364K</div>
</div>
<span class="schema-arrow">→</span>
<div style="text-align:center;">
<div class="schema-node secondary" style="min-width:70px;">R̂</div>
<div style="font-size:0.6em;color:#666;">Prédictions<br/>Dense</div>
</div>
</div>
</div>
</div>

</div>

<div>

## Avantages du SVD

<div class="success">

- **Réduit la dimensionnalité**
- Capture les **patterns latents**
- **Gère la sparsity** efficacement

</div>

### Hyperparamètres Optimaux

| Paramètre | Valeur |
|-----------|--------|
| n_factors | 50 |
| n_epochs | 30 |
| learning_rate | 0.01 |
| regularization | 0.02 |

</div>

</div>

---

# Comparaison SVD vs KNN

<div class="columns">

<div class="card" style="text-align: center;">

## SVD

<div class="metric">
<div class="metric-value">0.164</div>
<div class="metric-label">RMSE</div>
</div>

- **Plus précis** sur nos données
- Prédiction : **O(1)**
- Entraînement : ~5 min

</div>

<div class="card" style="text-align: center;">

## KNN (Cosine, k=40)

<div class="metric">
<div class="metric-value">0.180</div>
<div class="metric-label">RMSE</div>
</div>

- Interprétable
- Prédiction : **O(n)**
- Entraînement : ~1 min

</div>

</div>

<div class="success" style="margin-top: 15px;">

**Conclusion** : SVD est choisi comme modèle principal (RMSE **10% meilleur**)

</div>

---

<!-- _class: divider -->

# Approche Hybride

Combiner le meilleur des deux mondes

---

# Stratégie Hybride

<div class="columns">

<div>

## Weighted Hybrid

<div class="formula">

$$\text{score}_{final} = \alpha \times \text{score}_{CF} + (1-\alpha) \times \text{score}_{CB}$$

</div>

### Pondération Adaptative selon l'Activité

| Utilisateur | Clics | CB/CF | Stratégie CB |
|-------------|-------|-------|--------------|
| Nouveau | 1-2 | **80/20** | Last Click |
| Intermédiaire | 3-10 | **50/50** | Combined |
| Actif | >10 | **30/70** | Average |

<div class="concept" style="font-size:0.85em;">

**Plus l'historique est riche → plus le CF domine** (patterns collaboratifs fiables)

</div>

</div>

<div>

## Logique de Fallback

<div class="diagram" style="padding: 12px;">
<div class="decision">
<div class="schema-node primary">📥 Requete User ID</div>
<div class="schema-arrow">↓</div>
<div class="decision-node decision-diamond">User connu?</div>
<div style="display:flex; gap: 30px; margin-top: 5px;">
<div class="decision" style="gap:3px;">
<span class="decision-yes">OUI</span>
<div class="schema-arrow">↓</div>
<div class="schema-node secondary small">Hybrid (CF+CB)</div>
</div>
<div class="decision" style="gap:3px;">
<span class="decision-no">NON</span>
<div class="schema-arrow">↓</div>
<div class="decision-node decision-diamond" style="font-size:0.65em;">Articles lus?</div>
<div style="display:flex; gap:15px; margin-top:3px;">
<div style="text-align:center;">
<span class="decision-yes" style="font-size:0.6em;">OUI</span><br/>
<div class="schema-node accent small">CB seul</div>
</div>
<div style="text-align:center;">
<span class="decision-no" style="font-size:0.6em;">NON</span><br/>
<div class="schema-node danger small">Popular</div>
</div>
</div>
</div>
</div>
</div>
</div>

</div>

</div>

---

# Gestion du Cold-Start : CB vs CF

<div class="columns">

<div>

## Content-Based

<div class="card card-green">

### Nouveaux Utilisateurs
- **1 clic suffit** pour recommander
- Similarité immédiate via embeddings
- Pas de dépendance aux autres users

### Nouveaux Articles
- ✅ **Géré nativement**
- Embedding disponible dès la publication
- Recommandable immédiatement

</div>

</div>

<div>

## Collaborative Filtering

<div class="card card-orange">

### Nouveaux Utilisateurs
- ❌ **Non géré** - pas dans matrice P
- Fallback nécessaire (popularity)
- Nécessité retraining pour intégration

### Nouveaux Articles
- ❌ **Non géré** - pas dans matrice Q
- Invisible jusqu'au prochain batch
- Délai : 1 semaine (retraining)

</div>

</div>

</div>

<div class="warning">

**Stratégie Hybride** : CB prend le relais quand CF échoue (fallback automatique)

</div>

---

<!-- _class: divider -->

# Architecture Cloud

Du concept à la production

---

# Architecture MVP

<div class="columns">

<div>

## Composants Azure

| Service | Role |
|---------|------|
| **Functions** | API serverless (Python 3.11) |
| **Blob Storage** | Modèles et embeddings |
| **App Insights** | Monitoring et logs |

### Caractéristiques
- **Consumption Plan** : Pay-per-use
- **Latence moyenne** : < 200ms

</div>

<div>

## Flux de Données

<div class="diagram" style="padding: 15px;">
<div class="arch-diagram">
<div class="arch-layer" style="justify-content:center; background:linear-gradient(90deg,#fce4ec,#f8bbd9);">
<div class="arch-box client">📱 Web App</div>
<div class="arch-box client">📱 Mobile App</div>
</div>
<div class="arch-connector">↓ <span style="font-size:0.7em;">GET /recommendations/{user_id}</span> ↓</div>
<div class="arch-layer" style="background:linear-gradient(90deg,#fff3e0,#ffe0b2);">
<div class="arch-layer-icon">⚡</div>
<div class="arch-layer-title">API</div>
<div class="arch-components">
<div class="arch-box ingest">Azure Functions (Python 3.11)</div>
<div class="arch-box ingest">FastAPI + Surprise</div>
</div>
</div>
<div class="arch-connector">↓ Load models ↓</div>
<div class="arch-layer" style="background:linear-gradient(90deg,#e8f5e9,#c8e6c9);">
<div class="arch-layer-icon">💾</div>
<div class="arch-layer-title">Storage</div>
<div class="arch-components">
<div class="arch-box storage">📦 Blob: Models SVD</div>
<div class="arch-box storage">📦 Blob: Embeddings</div>
<div class="arch-box storage">📦 Blob: Ratings</div>
</div>
</div>
<div style="text-align:right; margin-top:8px;">
<div class="arch-box process" style="display:inline-flex;">📊 App Insights (Monitoring)</div>
</div>
</div>
</div>

</div>

</div>

---

# Architecture Cible (Production)

<div class="diagram" style="padding: 15px;">
<div class="arch-diagram">

<div class="arch-layer" style="background:linear-gradient(90deg,#fce4ec,#f8bbd9);">
<div class="arch-layer-icon">👥</div>
<div class="arch-layer-title">Clients</div>
<div class="arch-components">
<div class="arch-box client">📱 Mobile</div>
<div class="arch-box client">🌐 Web</div>
<div class="arch-box client">🔌 API Partenaires</div>
</div>
</div>

<div class="arch-connector">↕ HTTPS / WebSocket ↕</div>

<div class="arch-layer" style="background:linear-gradient(90deg,#e3f2fd,#bbdefb);">
<div class="arch-layer-icon">🚀</div>
<div class="arch-layer-title">Serving</div>
<div class="arch-components">
<div class="arch-box serve">🔒 API Management</div>
<div class="arch-box serve">⚡ Azure Functions</div>
<div class="arch-box serve">🌍 CDN (Assets)</div>
</div>
</div>

<div class="arch-connector">↕ Read Models ↕</div>

<div class="arch-layer" style="background:linear-gradient(90deg,#e8f5e9,#c8e6c9);">
<div class="arch-layer-icon">💾</div>
<div class="arch-layer-title">Storage</div>
<div class="arch-components">
<div class="arch-box storage">🗄️ Cosmos DB</div>
<div class="arch-box storage">📦 Blob Storage</div>
<div class="arch-box storage">⚡ Redis Cache</div>
</div>
</div>

<div class="arch-connector">↑ Deploy Models ↑</div>

<div class="arch-layer" style="background:linear-gradient(90deg,#f3e5f5,#e1bee7);">
<div class="arch-layer-icon">🧠</div>
<div class="arch-layer-title">Processing</div>
<div class="arch-components">
<div class="arch-box process">⚙️ Databricks</div>
<div class="arch-box process">🤖 Azure ML</div>
<div class="arch-box process">📊 MLflow</div>
</div>
</div>

<div class="arch-connector">↑ ETL Pipeline ↑</div>

<div class="arch-layer" style="background:linear-gradient(90deg,#fff3e0,#ffe0b2);">
<div class="arch-layer-icon">📥</div>
<div class="arch-layer-title">Ingestion</div>
<div class="arch-components">
<div class="arch-box ingest">🔄 Event Hub</div>
<div class="arch-box ingest">⚙️ Data Factory</div>
<div class="arch-box ingest">📋 Data Lake</div>
</div>
</div>

</div>
</div>

---

# Mise à Jour des Données

<div class="columns">

<div>

## Fréquences

| Composant | Fréquence | Trigger |
|-----------|-----------|---------|
| Profils | **Temps réel** | Nouveau clic |
| Cache Redis | **Horaire** | Scheduled |
| Embeddings | **Quotidien** | Nouveaux articles |
| Modèle SVD | **Hebdo** | Batch retrain |

</div>

<div>

## Pipeline de Retraining

<div class="diagram" style="padding: 10px;">
<div class="pipeline">
<div class="pipeline-step">
<div class="pipeline-num">1</div>
<div class="pipeline-box" style="background:#fff3e0; border:1px solid #ffb74d;">📥 Collecte</div>
</div>
<div class="pipeline-arrow">→</div>
<div class="pipeline-step">
<div class="pipeline-num">2</div>
<div class="pipeline-box" style="background:#fff3e0; border:1px solid #ffb74d;">⚙️ ETL</div>
</div>
<div class="pipeline-arrow">→</div>
<div class="pipeline-step">
<div class="pipeline-num">3</div>
<div class="pipeline-box" style="background:#f3e5f5; border:1px solid #ba68c8;">🧠 Train</div>
</div>
<div class="pipeline-arrow">→</div>
<div class="pipeline-step">
<div class="pipeline-num">4</div>
<div class="pipeline-box" style="background:#e8f5e9; border:1px solid #81c784;">✅ Valid</div>
</div>
<div class="pipeline-arrow">→</div>
<div class="pipeline-step">
<div class="pipeline-num">5</div>
<div class="pipeline-box" style="background:#e3f2fd; border:1px solid #64b5f6;">🚀 Deploy</div>
</div>
</div>
</div>

</div>

</div>

---

<!-- _class: divider -->

# Déploiement Azure

Infrastructure as Code avec Pulumi

---

# Stack Technique

<div class="columns-3">

<div class="card">

### Runtime
- **Python 3.11**
- FastAPI (async)
- Azure Functions v4
- Surprise (ML)

</div>

<div class="card">

### Infrastructure
- **Pulumi** (IaC)
- Blob Storage
- App Insights
- Consumption Plan

</div>

<div class="card">

### CI/CD
- **GitHub Actions**
- Tests automatisés
- Deploy on merge
- Health checks

</div>

</div>

<div class="concept">

**Infrastructure as Code** : Toute l'infrastructure définie en Python avec Pulumi

</div>

```python
function_app = web.WebApp("recommendation-api",
    site_config=web.SiteConfigArgs(linux_fx_version="PYTHON|3.11",
        app_settings=[{"name": "FUNCTIONS_WORKER_RUNTIME", "value": "python"}]))
```

---

# Optimisation Mémoire : Problème & Analyse PCA

<div class="columns">

<div>

## Contrainte

<div class="warning">

**Azure Functions Consumption Plan** : limite **1.5 GB** de RAM
Embeddings seuls : **347 MB** (250 dims) + overhead Python → dépasse la limite

</div>

## Solution : PCA

Réduire les embeddings de **250 → 50 dimensions** via Principal Component Analysis

![Scree Plot et Variance Cumulative](result/plots/05_pca_explained_variance.png)

</div>

<div>

## Validation Qualité

![Corrélation et Trade-off Mémoire vs Qualité](result/plots/06_memory_quality_tradeoff.png)

La corrélation des similarités cosinus atteint un **plateau à ~0.93 dès n=50**. Au-delà, aucun gain significatif.

## Résultat : n = 50

<div class="columns" style="gap:10px;">

<div class="card card-orange" style="font-size:0.85em;">

**Avant** : 250 dims, **347 MB**

</div>

<div class="card card-green" style="font-size:0.85em;">

**Après** : 50 dims, **69 MB** ✅

</div>

</div>

<div class="success">

**80% de réduction mémoire**, calcul **5x plus rapide**, **95% de variance** préservée

</div>

</div>

</div>

---

# API Endpoints

<div class="columns">

<div>

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /recommendations/{user_id}` | Recommendations |

### Paramètres

| Param | Default | Description |
|-------|---------|-------------|
| `n` | 5 | Nombre de recs |
| `method` | hybrid | collaborative, content, hybrid, popular |

</div>

<div>

## Exemple Réponse

```json
{
  "user_id": 12345,
  "recommendations": [
    157432, 289541, 94523,
    187234, 312456
  ],
  "method": "hybrid",
  "score": 0.847
}
```

### Codes Retour
- **200** : Succès (fallback auto si user inconnu)
- **500** : Erreur serveur

</div>

</div>

---

# Déploiement en Une Commande

<div class="columns">

<div>

## Script

```bash
./script/deploy.sh [dev|prod]
```

### Étapes automatisées
1. Vérifie prérequis (az, func, pulumi)
2. Login Azure et Pulumi
3. Deploy infrastructure
4. Upload modèles (si plus récents)
5. Publish Azure Functions
6. Health check automatique

</div>

<div>

## Output

```
==========================================
Recommendation System - Deployment
==========================================
Stack: dev

[deploy] Prerequisites... OK
[deploy] Azure login... OK
[deploy] Infrastructure... OK
[deploy] Models:
  Uploading svd_model.pkl (newer)
  Skipping embeddings.pkl (up-to-date)
  
[deploy] Health check PASSED (HTTP 200)
==========================================
URL: https://xxx.azurewebsites.net
```

</div>

</div>

---

<!-- _class: divider -->

# Résultats & Évaluation

Métriques et performances

---

# Problématique de l'Évaluation

<div class="columns">

<div>

## Limites des Données Offline

<div class="card card-red">

### Ce qu'on mesure
- **RMSE** : Erreur de prédiction des ratings
- **Hit Rate** : Article caché retrouvé

### Ce qu'on ne peut PAS mesurer
- **Satisfaction utilisateur** réelle
- **Engagement** (temps de lecture, partage)
- **Sérendipité** (découverte inattendue)
- **Diversité** des recommandations

</div>

</div>

<div>

## Recommandation : Tests A/B en Production

<div class="card card-green">

### Pourquoi tester en prod ?
- Les métriques offline ≠ succès réel
- Seul le **comportement utilisateur** valide le système
- Feedback implicite continu

### Métriques à suivre
- **CTR** (Click-Through Rate)
- **Temps de lecture** moyen
- **Taux de rebond**
- **Rétention utilisateur**

</div>

<div class="warning" style="margin-top:10px;">

**Conclusion** : Évaluation offline = point de départ, pas validation finale

</div>

</div>

</div>

---

# Métriques de Performance

<div class="columns">

<div>

## Collaborative Filtering

| Modèle | RMSE | MAE |
|--------|------|-----|
| **SVD** | **0.164** | **0.099** |
| KNN Pearson | 0.180 | 0.107 |
| KNN Cosine | 0.181 | 0.109 |

<div class="success">

**SVD surpasse KNN de ~10%** en RMSE

</div>

</div>

<div>

## Métriques Clés

<div class="columns" style="gap: 10px;">

<div class="metric card">
<div class="metric-value">0.164</div>
<div class="metric-label">RMSE</div>
</div>

<div class="metric card">
<div class="metric-value">0.099</div>
<div class="metric-label">MAE</div>
</div>

</div>

<div class="columns" style="gap: 10px; margin-top: 10px;">

<div class="metric card">
<div class="metric-value">&lt;200ms</div>
<div class="metric-label">Latence</div>
</div>

<div class="metric card">
<div class="metric-value">99.9%</div>
<div class="metric-label">Uptime</div>
</div>

</div>

</div>

</div>

---

<!-- _class: divider -->

# Conclusions

Bilan et perspectives

---

# Objectifs Atteints

<div class="columns">

<div>

## Réalisations

<div class="success">

- **Content-Based** avec embeddings + cosine
- **Collaborative** avec SVD (RMSE: 0.164)
- **Hybride** avec fallback intelligent
- **API Serverless** sur Azure Functions
- **IaC** avec Pulumi

</div>

</div>

<div>

## Métriques Finales

| Critère | Objectif | Résultat |
|---------|----------|----------|
| RMSE CF | < 0.20 | **0.164** |
| Latence | < 500ms | **< 200ms** |
| Cold-start | Géré | **100%** |
| Uptime | 99% | **99.9%** |

<div class="info">

**Tous les critères atteints**

</div>

</div>

</div>

---

# Limitations & Améliorations

<div class="columns">

<div>

## Limitations Actuelles

<div class="card card-red">

- **Batch updates** seulement
- **Single region** (France)
- **Pas de A/B testing**
- **ML classique** (pas de DL)

</div>

</div>

<div>

## Améliorations Futures

<div class="card card-green">

| Actuel | Futur |
|--------|-------|
| Batch | **Incremental** |
| Single region | **Multi-region** |
| Pas A/B | **Expérimentation** |
| SVD | **Transformers** |
| Manuel | **MLOps** |

</div>

</div>

</div>

---

<!-- _class: divider -->

# Questions ?

Merci pour votre attention !

---

# Annexe : Ressources

<div class="columns">

<div>

## Documentation
- [Surprise Library](https://surpriselib.com/)
- [Azure Functions](https://docs.microsoft.com/azure/azure-functions/)
- [Pulumi Azure](https://www.pulumi.com/docs/clouds/azure/)

## Code Source
- [Repository GitHub](https://github.com/sidux/oc-student-project-10)
- [API Azure](https://oc-student-project-10-func.azurewebsites.net)
- Notebooks d'exploration
- Scripts de déploiement

</div>

<div>

## Références
- Koren (2009) - Matrix Factorization
- Lops et al. (2011) - Content-based RS
- Burke (2002) - Hybrid RS

</div>

</div>
