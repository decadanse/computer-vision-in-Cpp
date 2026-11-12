#  ЛР3: только ноутбук `lab3_supir_mask.ipynb` на macOS (.venv, полная очистка)

**Принцип.** Всё живёт внутри одной папки проекта. Python-пакеты — в `.venv`, кэши и веса — в `.cache/`. Удалили папки — система чистая. C++ часть, `cmake`, `pybind11`, `pytest` здесь **не нужны**.

---

## 0. Структура папки

```
lab3/                              <- корень, все команды отсюда
├── lab3_supir_mask.ipynb
├── DIV2K_valid_HR/                <- эталоны (png/jpg), HR_DIR в ноутбуке
├── SUPIR/                         <- клон репозитория (только для реального запуска)
└── models/                        <- веса SDXL и SUPIR (только для реального запуска)
```

Перед каждой командой убедитесь: `pwd` показывает корень проекта.

## 1. Окружение

```bash
python3 -m venv .venv
cat > env_local.sh <<'EOF'
export PIP_CACHE_DIR=$PWD/.cache/pip
export TORCH_HOME=$PWD/.cache/torch
export HF_HOME=$PWD/.cache/hf
export MPLCONFIGDIR=$PWD/.cache/mpl
export JUPYTER_CONFIG_DIR=$PWD/.cache/jupyter/config
export JUPYTER_DATA_DIR=$PWD/.cache/jupyter/data
export JUPYTER_RUNTIME_DIR=$PWD/.cache/jupyter/runtime
export IPYTHONDIR=$PWD/.cache/ipython
export PYTHONDONTWRITEBYTECODE=1
EOF
source .venv/bin/activate
source env_local.sh
python -m pip install --upgrade pip
```

**В каждом новом терминале:** `source .venv/bin/activate && source env_local.sh` (из корня). Без `env_local.sh` кэши уйдут в `~`.

## 2. Пакеты

**Режим проверки каркаса (без нейросети):**

```bash
pip install numpy Pillow opencv-python pandas matplotlib jupyterlab
```

**Реальный запуск (SUPIR + LPIPS + CLIP-IQA), дополнительно:**

```bash
pip install torch torchvision lpips pyiqa omegaconf
git clone https://github.com/Fanghua-Yu/SUPIR SUPIR        # URL проверьте
grep -viE '^(triton|xformers|torch|torchvision|bitsandbytes|gradio|gradio_client|fastapi|pydantic|uvicorn)([=<>~! ]|$)' SUPIR/requirements.txt > .cache/req_mac.txt
pip install -r .cache/req_mac.txt
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Если `pip` падает на конкретном пине (`==`) под Python 3.9, уберите или ослабьте эту строку в `.cache/req_mac.txt`. Список «Ignored versions» вторичен, смотрите последние строки `ERROR`. Если нужен Python 3.10+, он должен быть в `.venv` (системный 3.9.6 не подойдёт).

Веса (SDXL, SUPIR-v0Q) кладите в `./models/` и укажите пути в `SDXL_CKPT` / `SUPIR_CKPT` в ячейке с параметрами. Иначе берутся пути из `SUPIR/options/SUPIR_v0.yaml`, они могут вести за пределы проекта.

## 3. Запуск ноутбука

```bash
python -m jupyter lab
```

Запускайте из корня и только так (не глобальным `jupyter`). Первой ячейкой проверьте:

```python
import sys; print(sys.executable)   # путь должен содержать .venv
```

Если путь другой, ядро чужое и пакетов не увидит: закройте Jupyter, активируйте `.venv`, запустите заново.

**Параметры (ячейка с `SUPIR_ROOT`, `HR_DIR`, …):**

| Режим | Значения |
|---|---|
| Проверка каркаса | `USE_FAKE=True`, `USE_LPIPS=False`, `USE_CLIPIQA=False`, `LIMIT=2`, `CROP=256` |
| Реальный запуск | `USE_FAKE=False`, `USE_LPIPS=True`, `USE_CLIPIQA=True`, пути весов заданы |

Ноутбук ищет SUPIR только в `./SUPIR` (или в самой текущей папке). Результаты пишутся в `./results/` (`results.csv`, графики, монтажи).

## 4. Сохраните результаты до очистки

Очистка удалит `results/`. Скопируйте нужное **вне** проекта:

```bash
mkdir -p ~/Desktop/lab3_report && cp -R results ~/Desktop/lab3_report/
pip freeze > ~/Desktop/lab3_report/requirements-lock.txt    # по желанию, пока .venv активно
```

---

## Пауза: выйти из .venv без удаления и вернуться позже

Выход из окружения **ничего не удаляет**. `.venv`, `.cache/`, `results/`, `models/`, `SUPIR/` остаются на диске, пока вы сами не выполните раздел 5.

**Выйти (сделать паузу):**

```bash
# 1. остановите Jupyter: Ctrl+C в окне терминала, где он запущен (подтвердите y)
deactivate            # выход из .venv; приглашение «(.venv)» исчезнет
```

Закрытие окна терминала даёт тот же эффект: окружение «выключается», файлы остаются.

**Вернуться через несколько дней:**

```bash
cd /путь/к/lab3                           # корень проекта (там лежат .venv и env_local.sh)
source .venv/bin/activate && source env_local.sh
python -c "import sys; print(sys.executable)"    # путь должен содержать .venv
python -m jupyter lab
```

Заново ничего устанавливать не нужно. `env_local.sh` обязателен в каждом новом терминале: без него кэши снова уйдут в `~` (см. раздел 1).

**Что сохраняется, а что нет:**

| Сохраняется на диске | Пропадает при закрытии |
|---|---|
| `.venv` (все пакеты), `.cache/`, `models/`, `SUPIR/`, `DIV2K_valid_HR/` | переменные оболочки (`PIP_CACHE_DIR` и др., поэтому нужен `source env_local.sh`) |
| `results/` (`results.csv`, графики, монтажи) | запущенный Jupyter и **переменные в памяти ядра**: после возврата выполните ячейки заново сверху вниз |
| сохранённый `.ipynb` (с выводами, если вы нажали Save) | несохранённые правки ноутбука |

**Осторожно, чтобы `.venv` не сломался за время паузы:**
- Не переименовывайте и не переносите папку проекта: внутри `.venv` записаны абсолютные пути, после переноса окружение перестаёт работать. Если пришлось, удалите `.venv` и создайте заново (раздел 1–2).
- Обновление macOS или Xcode Command Line Tools может заменить системный Python 3.9.6, на который ссылается `.venv`. Признак: `python: command not found` или ошибки при `source .venv/bin/activate`. Попробуйте `python3 -m venv --upgrade .venv`; если не помогло, пересоздайте окружение. Это предположение о типичном поведении, я его на вашем Mac не проверял.
- Следите за местом: `du -sh .venv .cache models SUPIR` покажет, сколько занято. Веса моделей могут занимать десятки ГБ.

**Сохранить состояние на случай пересоздания (по желанию):**

```bash
pip freeze > requirements-lock.txt        # пока .venv активно; лежит в корне проекта
# позже: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-lock.txt
```

Когда закончите работу (через несколько дней), выполните раздел 4 (сохранить результаты вне проекта) и раздел 5 (полная очистка).

---

## 5. Полная очистка

Закройте Jupyter (`Ctrl+C` в терминале). Затем из корня проекта:

```bash
pwd                                   # убедитесь, что это корень проекта
python -m pip cache purge             # пока .venv активно
deactivate
ls -d .venv .cache SUPIR models results env_local.sh 2>/dev/null
rm -rf .venv .cache SUPIR models results env_local.sh .ipynb_checkpoints
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
```

`rm -rf` необратим: сверьтесь с выводом `ls`. Остаются только `lab3_supir_mask.ipynb` и `DIV2K_valid_HR/` (датасет удаляйте отдельно, если он больше не нужен). Ноутбук после работы содержит выводы ячеек: при желании очистите их через меню Jupyter *Edit → Clear Outputs of All Cells* до закрытия.

### Если что-то запускали без `env_local.sh`

| Что | Где | Удаление |
|---|---|---|
| Кэш pip | `~/Library/Caches/pip` | `rm -rf ~/Library/Caches/pip` |
| Кэш и ядра Jupyter | `~/Library/Jupyter`, `~/.jupyter`, `~/.ipython` | `rm -rf` (только если не используете Jupyter для другого) |
| Веса lpips/pyiqa/HF | `~/.cache/torch`, `~/.cache/huggingface`, `~/.cache/lpips` | `rm -rf` (проверьте `ls`: папки могут принадлежать другим программам) |
| Веса SUPIR вне проекта | пути из `SUPIR_v0.yaml` | `ls`, затем `rm -rf` |
| Переменные оболочки | текущий терминал | пропадают при закрытии терминала |

### Проверка

```bash
which python3                         # /usr/bin/python3
python3 -c "import numpy" 2>&1        # ModuleNotFoundError = системный Python чист
ls -d .venv .cache SUPIR models 2>&1  # No such file or directory
du -sh ~/Library/Caches/pip 2>&1      # No such file или маленький размер
```

Если `import numpy` в системном Python работает, он стоял там до вас, его не трогайте.

## Повторный запуск с нуля

Повторите разделы 1–3.
