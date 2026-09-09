# Push PixelForge to github.com/atkialamisha

The cloud agent cannot create repositories on the **atkialamisha** account (token is scoped to `atik-efaz5/PixelForge` only). Run these steps **on your Mac** while logged in as **atkialamisha**:

## 1. Create the repo (one time)

```bash
gh auth login   # choose GitHub.com → atkialamisha
gh repo create PixelForge --public \
  --description "SAM 2 + Moebius image editing — live demo on Vercel" \
  --homepage "https://frontend-mu-two-wzuqjziue7.vercel.app"
```

## 2. Push this project

```bash
cd ~/Projects/PixelForge   # or your clone path
git remote add atkialamisha https://github.com/atkialamisha/PixelForge.git 2>/dev/null || true
git push -u atkialamisha main
```

## 3. Set GitHub homepage (optional)

Repo → **Settings** → **General** → **Website** →  
`https://frontend-mu-two-wzuqjziue7.vercel.app`

## Alternative: transfer from atik-efaz5

If the code already lives at `atik-efaz5/PixelForge`, transfer ownership:

GitHub → `atik-efaz5/PixelForge` → **Settings** → **Danger Zone** → **Transfer ownership** → `atkialamisha`
