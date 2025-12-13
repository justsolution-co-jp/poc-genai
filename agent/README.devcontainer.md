# Devcontainer (安定優先)

このリポジトリはホスト(GX10)を汚染しないため、Devcontainer内で依存を導入します。

## 方針
- OSパッケージの `apt upgrade` は原則しません（再現性が落ちるため）。
- Python依存は `requirements.txt` を固定して再現性を担保します。
- 依存更新は「意図したタイミング」で行います。

## キャッシュ（Docker volume）
`hf-cache`（Hugging Face）と `pip-cache`（pip）は Docker volume に保存します。ホスト側のディレクトリ作成が不要で、Devcontainerの作り直しでもキャッシュが残ります。

確認:

```bash
docker volume ls | grep -E 'hf-cache|pip-cache'
docker volume inspect hf-cache
docker volume inspect pip-cache
```

初期化（注意: キャッシュ消えます）:

```bash
docker volume rm hf-cache pip-cache
```

## 依存の更新手順（手動）
1. `requirements.in` を編集
2. ロック生成:

```bash
python -m pip install -U pip pip-tools
pip-compile --generate-hashes --resolver=backtracking -o requirements.txt requirements.in
```

3. DevcontainerをRebuild

## よくある確認
```bash
python -V
pip -V
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## トラブルシュート
- `scipy` / `sentencepiece` がビルドに回る場合は、wheelがある組み合わせに寄せるか、必要に応じて `apt-get install -y build-essential` 等を追加してください（安定優先なら、まずはwheelで入る版に固定推奨）。
