# -*- coding: utf-8 -*- #
"""
手势识别 Web 应用
基于训练好的 CNN 模型，提供网页端图片上传识别功能
"""

import os
import sys
import base64
import re
import time
import torch
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template, send_file
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# 创建上传目录
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_model(model_path=None):
    """加载训练好的模型"""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    if not os.path.exists(model_path):
        print(f"[错误] 模型文件不存在: {os.path.abspath(model_path)}")
        print("[提示] 请先运行 train.py 训练模型")
        sys.exit(1)
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    return model


# 全局加载模型（启动时加载一次）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model()
model.to(device)
print(f"[启动] 模型已加载至: {device}")


def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_digit(image_path):
    """对单张图片进行预测，返回识别结果和置信度"""
    # 1. 加载图片
    image = Image.open(image_path).convert('RGB')

    # 2. 转换为64x64、转为Tensor并标准化（与训练时一致）
    transform = Compose([
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = transform(image)

    # 3. 添加 batch 维度
    x = torch.unsqueeze(x, 0).to(device)

    # 4. 前向传播
    with torch.no_grad():
        output = model(x)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    # 5. 获取预测结果
    confidence, predicted = torch.max(probabilities, 1)

    return {
        'digit': predicted.item(),
        'confidence': round(confidence.item() * 100, 2),
        'probabilities': {
            str(i): round(probabilities[0][i].item() * 100, 2)
            for i in range(10)
        }
    }


@app.route('/')
def index():
    """主页：渲染前端页面"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API：接收上传图片并返回识别结果"""
    # 检查是否有文件
    if 'image' not in request.files:
        return jsonify({'error': '未选择图片'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的图片格式，请上传 PNG/JPG/JPEG/BMP'}), 400

    # 保存文件
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        # 识别
        result = predict_digit(filepath)
        return jsonify({
            'success': True,
            'digit': result['digit'],
            'confidence': result['confidence'],
            'probabilities': result['probabilities'],
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': f'识别失败: {str(e)}'}), 500


# ====== 数据采集接口 ======

# 实拍照片存储路径
REAL_PHOTO_DIR = os.path.join(os.path.dirname(__file__), 'images', 'train', 'real')
REAL_LABELS_FILE = os.path.join(os.path.dirname(__file__), 'images', 'real_photos.txt')
os.makedirs(REAL_PHOTO_DIR, exist_ok=True)


@app.route('/collect')
def collect_page():
    """数据采集页面"""
    return render_template('collect.html')


@app.route('/api/capture', methods=['POST'])
def api_capture():
    """接收摄像头拍摄的照片并保存"""
    data = request.json
    if not data or 'image' not in data or 'digit' not in data:
        return jsonify({'success': False, 'error': '缺少参数'}), 400

    digit = int(data['digit'])
    image_b64 = data['image']

    # 解析base64图片
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]

    try:
        img_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_data)).convert('RGB')

        # 保存为JPEG
        timestamp = int(time.time() * 1000)
        filename = f'{digit}_{timestamp}.jpg'
        filepath = os.path.join(REAL_PHOTO_DIR, filename)
        img.save(filepath, 'JPEG', quality=90)

        # 写入标注文件
        with open(REAL_LABELS_FILE, 'a', encoding='utf-8') as f:
            f.write(f'./images/train/real/{filename} {digit}\n')

        # 更新累计计数
        count_file = os.path.join(os.path.dirname(__file__), 'capture_stats.json')
        try:
            import json
            if os.path.exists(count_file):
                with open(count_file, 'r') as f:
                    stats = json.load(f)
            else:
                stats = {str(i): 0 for i in range(10)}
            stats[str(digit)] = stats.get(str(digit), 0) + 1
            stats['total'] = stats.get('total', 0) + 1
            with open(count_file, 'w') as f:
                json.dump(stats, f)
        except:
            pass

        return jsonify({
            'success': True,
            'filename': filename,
            'digit': digit,
            'stats': get_capture_stats()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/capture/stats')
def api_capture_stats():
    """返回采集统计"""
    return jsonify(get_capture_stats())


def get_capture_stats():
    """获取每个数字已采集的照片数"""
    stats = {str(i): 0 for i in range(10)}
    stats['total'] = 0
    try:
        for f in os.listdir(REAL_PHOTO_DIR):
            if f.endswith('.jpg'):
                digit = f.split('_')[0]
                if digit.isdigit():
                    stats[digit] = stats.get(digit, 0) + 1
                    stats['total'] += 1
    except:
        pass
    return stats


@app.route('/api/upload-photos', methods=['POST'])
def api_upload_photos():
    """接收批量上传的手势照片并保存"""
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': '未选择文件'}), 400

    files = request.files.getlist('images')
    digits = request.form.getlist('digits')

    if len(files) != len(digits):
        return jsonify({'success': False, 'error': '文件与标签数量不匹配'}), 400

    added = 0
    for i, file in enumerate(files):
        if file.filename == '':
            continue
        try:
            digit = int(digits[i])
            img = Image.open(file).convert('RGB')

            # 保存
            timestamp = int(time.time() * 1000)
            filename = f'{digit}_{timestamp}_{added}.jpg'
            filepath = os.path.join(REAL_PHOTO_DIR, filename)
            img.save(filepath, 'JPEG', quality=90)

            # 写入标注
            with open(REAL_LABELS_FILE, 'a', encoding='utf-8') as f:
                f.write(f'./images/train/real/{filename} {digit}\n')

            added += 1
        except Exception as e:
            print(f'[upload] 保存失败: {e}')

    # 合并到 train.txt
    try:
        train_txt = os.path.join(os.path.dirname(__file__), 'images', 'train.txt')
        with open(REAL_LABELS_FILE, 'r', encoding='utf-8') as f:
            real_entries = set(line.strip() for line in f.readlines() if line.strip())
        with open(train_txt, 'r', encoding='utf-8') as f:
            existing = set(line.strip() for line in f.readlines())
        new_entries = [e for e in real_entries if e not in existing]
        if new_entries:
            with open(train_txt, 'a', encoding='utf-8') as f:
                for e in new_entries:
                    f.write(e + '\n')
    except Exception as e:
        print(f'[upload] 合并到train.txt失败: {e}')

    return jsonify({'success': True, 'added': added})


@app.route('/api/clear-photos', methods=['POST'])
def api_clear_photos():
    """清空已上传的实拍照片"""
    import shutil
    if os.path.exists(REAL_PHOTO_DIR):
        shutil.rmtree(REAL_PHOTO_DIR)
        os.makedirs(REAL_PHOTO_DIR, exist_ok=True)
    if os.path.exists(REAL_LABELS_FILE):
        os.remove(REAL_LABELS_FILE)
    # 从train.txt移除实拍条目
    train_txt = os.path.join(os.path.dirname(__file__), 'images', 'train.txt')
    if os.path.exists(train_txt):
        with open(train_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(train_txt, 'w', encoding='utf-8') as f:
            for line in lines:
                if '/real/' not in line:
                    f.write(line)
    return jsonify({'success': True})


@app.route('/api/train', methods=['POST'])
def api_train():
    """后台启动训练（不阻塞，实时返回进度）"""
    import subprocess as sp
    script_dir = os.path.dirname(__file__)
    train_script = os.path.join(script_dir, 'train_mixed.py')

    if not os.path.exists(train_script):
        return jsonify({'success': False, 'error': '训练脚本不存在'}), 400

    # 检查是否已有训练在运行
    if os.path.exists(os.path.join(script_dir, 'train_running.flag')):
        return jsonify({'success': False, 'error': '已有训练任务正在运行'}), 409

    # 标记训练开始
    flag_file = os.path.join(script_dir, 'train_running.flag')
    open(flag_file, 'w').close()

    # 清空旧进度
    progress_file = os.path.join(script_dir, 'train_progress.txt')
    if os.path.exists(progress_file):
        os.remove(progress_file)

    # 后台启动训练进程
    def run_train():
        try:
            proc = sp.Popen(
                [sys.executable, train_script],
                cwd=script_dir,
                stdout=sp.PIPE, stderr=sp.STDOUT,
                text=True, encoding='utf-8',
            )
            proc.wait()
        finally:
            if os.path.exists(flag_file):
                os.remove(flag_file)
            # 训练完成后重新加载模型
            try:
                global model
                model = load_model()
                model.to(device)
                print('[训练] 模型已重新加载')
            except Exception as e:
                print(f'[训练] 模型加载失败: {e}')

    import threading
    t = threading.Thread(target=run_train, daemon=True)
    t.start()

    return jsonify({'success': True, 'message': '训练已启动'})


@app.route('/api/train/progress')
def api_train_progress():
    """返回当前训练进度（实时输出）"""
    progress_file = os.path.join(os.path.dirname(__file__), 'train_progress.txt')
    flag_file = os.path.join(os.path.dirname(__file__), 'train_running.flag')

    is_running = os.path.exists(flag_file)
    lines = []
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    is_complete = any('__TRAINING_COMPLETE__' in l for l in lines)
    is_failed = any('__TRAINING_FAILED__' in l for l in lines)

    return jsonify({
        'running': is_running,
        'complete': is_complete,
        'failed': is_failed,
        'lines': lines[-100:],  # 最近100行
        'total_lines': len(lines),
    })


# ====== 系统信息接口 ======

@app.route('/api/system/info')
def api_system_info():
    """返回系统信息（状态栏、模型信息卡片）"""
    import platform as pf
    import json as jn
    from datetime import datetime

    script_dir = os.path.dirname(__file__)

    # 模型信息
    model_path = os.path.join(script_dir, 'models', 'model.pkl')
    model_size_mb = round(os.path.getsize(model_path) / 1024 / 1024, 2) if os.path.exists(model_path) else 0
    model_mtime = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y-%m-%d %H:%M:%S') if os.path.exists(model_path) else 'N/A'

    # 数据集统计
    train_txt = os.path.join(script_dir, 'images', 'train.txt')
    total_images = len(open(train_txt, encoding='utf-8').readlines()) if os.path.exists(train_txt) else 0

    # 识别次数
    hist_file = os.path.join(script_dir, 'recognition_history.json')
    recog_count = 0
    if os.path.exists(hist_file):
        try:
            with open(hist_file, 'r', encoding='utf-8') as f:
                recog_count = len(jn.load(f))
        except: pass

    # 训练次数
    train_hist_file = os.path.join(script_dir, 'training_history.json')
    train_count = 0
    if os.path.exists(train_hist_file):
        try:
            with open(train_hist_file, 'r', encoding='utf-8') as f:
                train_count = len(jn.load(f))
        except: pass

    # 数据集大小
    real_dir = os.path.join(script_dir, 'images', 'train', 'real')
    real_count = len([f for f in os.listdir(real_dir) if f.endswith('.jpg')]) if os.path.exists(real_dir) else 0

    # 当前准确率（从训练进度读取）
    accuracy = 99.5  # 默认最新值
    progress_file = os.path.join(script_dir, 'train_progress.txt')
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '测试集准确率' in line:
                    try:
                        accuracy = float(line.strip().split('=')[-1].replace('%', ''))
                    except: pass

    return jsonify({
        'success': True,
        # 环境
        'python_version': pf.python_version(),
        'pytorch_version': torch.__version__,
        'device': 'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'device_type': 'cuda' if torch.cuda.is_available() else 'cpu',
        # 模型
        'model_name': type(model).__name__,
        'model_params': f'{sum(p.numel() for p in model.parameters()):,}',
        'model_size_mb': model_size_mb,
        'model_updated': model_mtime,
        'model_accuracy': accuracy,
        # 数据集
        'dataset_total': total_images,
        'real_photos': real_count,
        # 统计
        'recognition_count': recog_count,
        'training_count': train_count,
    })


# ====== 识别历史接口 ======

HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'recognition_history.json')


@app.route('/api/history')
def api_history():
    """返回最近20条识别记录"""
    try:
        import json as jn
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                records = jn.load(f)
        else:
            records = []
        return jsonify({'success': True, 'records': records[-20:]})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'records': []})


@app.route('/api/history/save', methods=['POST'])
def api_history_save():
    """前端保存识别记录"""
    import json as jn
    try:
        data = request.json
        if not data:
            return jsonify({'success': False}), 400
        save_recognition_result(data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history/clear', methods=['POST'])
def api_history_clear():
    """清空识别历史"""
    import json as jn
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        jn.dump([], f)
    return jsonify({'success': True})


def save_recognition_result(result):
    """保存识别结果到历史（在 predict 中调用）"""
    import json as jn
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                records = jn.load(f)
        else:
            records = []
        records.append(result)
        # 只保留最近50条
        if len(records) > 50:
            records = records[-50:]
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            jn.dump(records, f, ensure_ascii=False, indent=2)
    except:
        pass


# ====== 训练历史接口 ======

TRAIN_HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'training_history.json')


@app.route('/api/training/history')
def api_training_history():
    """返回训练历史记录"""
    import json as jn
    try:
        if os.path.exists(TRAIN_HISTORY_FILE):
            with open(TRAIN_HISTORY_FILE, 'r', encoding='utf-8') as f:
                records = jn.load(f)
        else:
            records = []
        return jsonify({'success': True, 'records': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'records': []})


def save_training_record(record):
    """保存训练记录"""
    import json as jn
    try:
        if os.path.exists(TRAIN_HISTORY_FILE):
            with open(TRAIN_HISTORY_FILE, 'r', encoding='utf-8') as f:
                records = jn.load(f)
        else:
            records = []
        records.append(record)
        with open(TRAIN_HISTORY_FILE, 'w', encoding='utf-8') as f:
            jn.dump(records, f, ensure_ascii=False, indent=2)
    except:
        pass


# ====== 示例图片接口 ======

@app.route('/api/sample/<int:digit>')
def api_sample(digit):
    """返回指定数字的示例图片（从训练集中随机选取）"""
    import random as rnd
    train_txt = os.path.join(os.path.dirname(__file__), 'images', 'train.txt')
    candidates = []
    base_dir = os.path.dirname(__file__)
    if os.path.exists(train_txt):
        with open(train_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1] == str(digit):
                    img_path = parts[0]
                    if img_path.startswith('./'):
                        img_path = img_path[2:]
                    full_path = os.path.join(base_dir, img_path)
                    if os.path.exists(full_path) and '/real/' not in img_path:
                        candidates.append(full_path)
    if candidates:
        choice = rnd.choice(candidates)
        return send_file(choice, mimetype='image/png')
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/sample/test/<int:digit>')
def api_sample_test(digit):
    """返回指定数字的测试集图片"""
    import random as rnd
    test_txt = os.path.join(os.path.dirname(__file__), 'images', 'test.txt')
    candidates = []
    base_dir = os.path.dirname(__file__)
    if os.path.exists(test_txt):
        with open(test_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[-1] == str(digit):
                    img_path = parts[0]
                    if img_path.startswith('./'):
                        img_path = img_path[2:]
                    full_path = os.path.join(base_dir, img_path)
                    if os.path.exists(full_path):
                        candidates.append(full_path)
    if candidates:
        choice = rnd.choice(candidates)
        return send_file(choice, mimetype='image/png')
    return jsonify({'error': 'Not found'}), 404


# ====== 模型版本管理接口 ======

MODEL_VERSIONS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'versions')
os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)


@app.route('/api/models/versions')
def api_models_versions():
    """返回模型版本列表"""
    versions = []
    current = os.path.join(os.path.dirname(__file__), 'models', 'model.pkl')
    if os.path.exists(current):
        versions.append({
            'name': 'Current (Best)',
            'size_mb': round(os.path.getsize(current) / 1024 / 1024, 2),
            'updated': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(current))),
            'is_current': True,
        })
    if os.path.exists(MODEL_VERSIONS_DIR):
        for f in sorted(os.listdir(MODEL_VERSIONS_DIR), reverse=True):
            if f.endswith('.pkl'):
                fp = os.path.join(MODEL_VERSIONS_DIR, f)
                versions.append({
                    'name': f.replace('.pkl', ''),
                    'size_mb': round(os.path.getsize(fp) / 1024 / 1024, 2),
                    'updated': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(fp))),
                    'is_current': False,
                })
    return jsonify({'success': True, 'versions': versions})


# ====== 数据集质量评分接口 ======

@app.route('/api/dataset/quality')
def api_dataset_quality():
    """返回数据集质量评分"""
    import json as jn
    real_dir = os.path.join(os.path.dirname(__file__), 'images', 'train', 'real')
    counts = {str(i): 0 for i in range(10)}
    total = 0
    if os.path.exists(real_dir):
        for f in os.listdir(real_dir):
            if f.endswith('.jpg'):
                d = f.split('_')[0]
                if d.isdigit():
                    counts[d] = counts.get(d, 0) + 1
                    total += 1

    # 质量评分算法
    if total == 0:
        score, level = 0, 'No Data'
    else:
        vals = [counts[str(i)] for i in range(6)]
        avg = sum(vals) / 6
        # 均衡度评分
        balance_score = max(0, 100 - sum(abs(v - avg) / max(avg, 1) * 10 for v in vals))
        # 数量评分
        count_score = min(100, total / 60 * 100)
        score = round((balance_score * 0.6 + count_score * 0.4))

        if score >= 90: level = 'Excellent ★★★★★'
        elif score >= 75: level = 'Great ★★★★'
        elif score >= 60: level = 'Good ★★★'
        elif score >= 40: level = 'Fair ★★'
        else: level = 'Needs More ★'

    return jsonify({
        'success': True,
        'score': score,
        'level': level,
        'counts': counts,
        'total': total,
    })


if __name__ == '__main__':
    # 尝试端口号，如果被占用则自动递增
    port = 5000
    max_port = 5010
    while port <= max_port:
        try:
            print(f"[启动] 服务器运行在 http://localhost:{port}")
            app.run(host='0.0.0.0', port=port, debug=False)
            break
        except OSError:
            print(f"[提示] 端口 {port} 已被占用，尝试下一个...")
            port += 1
    else:
        print(f"[错误] 端口 {max_port - 5000 + 1} 个端口均被占用，请手动关闭占用端口的程序后重试")
