"""
SocialSim - AI社会模拟平台 v2.1
"""

import os
import json
import time
import uuid
import re
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from openai import OpenAI
import threading
import queue

app = Flask(__name__)

# ============================================
# 全局状态管理
# ============================================
class SimulationState:
    def __init__(self):
        self.world = {}
        self.agents = []
        self.history = []
        self.running = False
        self.speed = 3
        self.round = 0
        self.event_queue = queue.Queue()
        self.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
        self.model = 'qwen-plus'
        self.lock = threading.Lock()
        self.custom_templates = {}
        self.metrics = []
        self.metric_data = {}
        
state = SimulationState()

# ============================================
# Qwen API 调用
# ============================================
def call_qwen_api(messages, temperature=0.85):
    if not state.api_key:
        raise ValueError("请先设置API Key")
    
    client = OpenAI(
        api_key=state.api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    completion = client.chat.completions.create(
        model=state.model,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,
        top_p=0.9,
    )
    
    return completion.choices[0].message.content

# ============================================
# AI 生成角色
# ============================================
def generate_agents_for_world(world, count=4):
    """根据世界设定生成匹配的角色"""
    prompt = f"""你是一个社会模拟实验设计专家。请根据以下世界设定，生成{count}个适合这个世界的角色。

## 世界设定
【名称】{world.get('name', '未命名世界')}
【背景】{world.get('background', '')}
【规则】{world.get('rules', '')}
【资源】{world.get('resources', '')}

## 要求
生成{count}个角色，每个角色应该：
1. 有独特的性格特征和行为模式
2. 有明确的目标和动机
3. 有与世界设定相符的背景故事
4. 角色之间应该有潜在的互动关系（合作、竞争、冲突等）

请直接返回JSON数组格式，不要有其他内容：
[
  {{
    "name": "角色名称",
    "personality": "详细的性格描述（50-100字）",
    "goal": "角色的核心目标（30-50字）",
    "memory": "背景记忆和经历（30-50字）"
  }},
  ...
]"""

    try:
        messages = [{"role": "user", "content": prompt}]
        response = call_qwen_api(messages, temperature=0.8)
        
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            agents = json.loads(json_match.group())
            for agent in agents:
                agent['id'] = str(uuid.uuid4())
            return agents
    except Exception as e:
        print(f"生成角色失败: {e}")
    
    return []

# ============================================
# Prompt 构建器
# ============================================
def build_system_prompt(world):
    return f"""你是一个社会模拟实验的参与者。你需要完全沉浸在分配给你的角色中，根据角色的性格、目标和当前情境做出真实自然的反应。

## 世界设定
【世界名称】{world.get('name', '未命名世界')}
【背景描述】{world.get('background', '')}
【运行规则】{world.get('rules', '')}
【资源要素】{world.get('resources', '')}

## 行为准则
1. 始终保持角色一致性，你的一切言行都要符合角色的性格特征
2. 根据世界规则行事，不要超出设定的边界
3. 与其他角色自然互动，建立真实的社会关系
4. 朝着角色目标努力，但要符合逻辑和情境
5. 记住之前发生的事，保持记忆连续性"""

def build_agent_prompt(agent, all_agents, history, event_context=''):
    other_agents = [a for a in all_agents if a['id'] != agent['id']]
    recent_history = history[-20:] if len(history) > 20 else history
    
    history_text = '\n'.join([
        f"[回合{h['round']}] {h['agent']}: {h['content']}" 
        for h in recent_history
    ]) if recent_history else "（这是模拟的开始，还没有发生任何事情）"
    
    other_agents_text = '\n'.join([
        f"• {a['name']}: {a.get('personality', '未知')[:60]}..."
        for a in other_agents
    ]) if other_agents else "（目前没有其他角色）"
    
    prompt = f"""## 你的角色档案
【姓名】{agent['name']}
【性格特征】{agent.get('personality', '未设定')}
【核心目标】{agent.get('goal', '未设定')}
【背景记忆】{agent.get('memory', '未设定')}

## 世界中的其他角色
{other_agents_text}

## 最近发生的事（按时间顺序）
{history_text}

{f"## ⚡ 突发事件{chr(10)}{event_context}{chr(10)}" if event_context else ""}

## 现在轮到你行动
请以 {agent['name']} 的身份，根据你的性格和目标，自然地做出反应。

输出格式说明：
- 用 *星号* 包裹动作描述
- 用 "引号" 包裹说出的话
- 用 (括号) 包裹内心想法
- 可以自由组合以上元素

要求：
1. 回应最近发生的事，保持对话连贯性
2. 展现角色独特的说话方式和行为风格
3. 适当推进你的目标，但不要太刻意
4. 保持真实感，像真人一样有情绪波动
5. 回复长度适中（50-150字），不要太短也不要太长"""

    return prompt

def build_metric_analysis_prompt(metrics, history, round_num):
    recent_history = history[-10:] if len(history) > 10 else history
    history_text = '\n'.join([
        f"[{h['agent']}]: {h['content']}" 
        for h in recent_history
    ])
    
    metrics_desc = '\n'.join([
        f"- {m['name']}: {m['description']} (范围: {m.get('min', 0)}-{m.get('max', 100)})"
        for m in metrics
    ])
    
    return f"""你是一个社会模拟实验的观察者。请根据最近发生的事件，评估以下指标的当前值。

## 需要评估的指标
{metrics_desc}

## 最近发生的事件
{history_text}

## 任务
请为每个指标给出一个数值评估。直接返回JSON格式，不要有其他内容：
{{"指标名称": 数值, ...}}

注意：
1. 数值必须在指定范围内
2. 根据事件内容合理推断
3. 只返回JSON，不要解释"""

# ============================================
# 指标分析
# ============================================
def analyze_metrics():
    if not state.metrics or not state.history:
        return
    
    try:
        messages = [
            {"role": "user", "content": build_metric_analysis_prompt(
                state.metrics, state.history, state.round
            )}
        ]
        
        response = call_qwen_api(messages, temperature=0.3)
        
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            values = json.loads(json_match.group())
            
            for metric in state.metrics:
                metric_name = metric['name']
                if metric_name in values:
                    value = float(values[metric_name])
                    value = max(metric.get('min', 0), min(metric.get('max', 100), value))
                    
                    if metric['id'] not in state.metric_data:
                        state.metric_data[metric['id']] = []
                    
                    state.metric_data[metric['id']].append({
                        'round': state.round,
                        'value': value
                    })
    except Exception as e:
        print(f"指标分析失败: {e}")

# ============================================
# 模拟引擎
# ============================================
def run_simulation_step():
    with state.lock:
        if not state.agents:
            return None
        
        state.round += 1
        
        event_context = ''
        try:
            event_context = state.event_queue.get_nowait()
        except queue.Empty:
            pass
        
        agent_index = (state.round - 1) % len(state.agents)
        current_agent = state.agents[agent_index]
        
        messages = [
            {"role": "system", "content": build_system_prompt(state.world)},
            {"role": "user", "content": build_agent_prompt(
                current_agent, 
                state.agents, 
                state.history,
                event_context
            )}
        ]
        
        try:
            response = call_qwen_api(messages)
            
            log_entry = {
                'id': str(uuid.uuid4()),
                'round': state.round,
                'agent': current_agent['name'],
                'agent_id': current_agent['id'],
                'content': response,
                'timestamp': datetime.now().isoformat(),
                'event': event_context if event_context else None
            }
            state.history.append(log_entry)
            
            if state.metrics and state.round % 5 == 0:
                analyze_metrics()
            
            return log_entry
            
        except Exception as e:
            error_entry = {
                'id': str(uuid.uuid4()),
                'round': state.round,
                'agent': 'System',
                'agent_id': 'system',
                'content': f'❌ API调用失败: {str(e)}',
                'timestamp': datetime.now().isoformat(),
                'error': True
            }
            state.history.append(error_entry)
            return error_entry

def simulation_loop():
    while state.running:
        result = run_simulation_step()
        if result and result.get('error'):
            state.running = False
            break
        time.sleep(state.speed)

# ============================================
# API 路由
# ============================================
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'POST':
        data = request.json
        if 'api_key' in data:
            state.api_key = data['api_key']
        if 'model' in data:
            state.model = data['model']
        return jsonify({'success': True})
    else:
        return jsonify({
            'has_key': bool(state.api_key),
            'model': state.model
        })

@app.route('/api/world', methods=['GET', 'POST'])
def world():
    if request.method == 'POST':
        state.world = request.json
        return jsonify({'success': True, 'message': '世界设定已保存'})
    else:
        return jsonify(state.world)

@app.route('/api/templates', methods=['GET'])
def get_templates():
    all_templates = {**TEMPLATES, **state.custom_templates}
    return jsonify(all_templates)

@app.route('/api/templates/save', methods=['POST'])
def save_template():
    """保存当前设定为新模板"""
    data = request.json
    template_name = data.get('name', '').strip()
    template_id = data.get('id')  
    auto_generate = data.get('auto_generate', False)
    
    if not template_name:
        return jsonify({'success': False, 'message': '请输入模板名称'}), 400
    
    if not state.world.get('name') and not state.world.get('background'):
        return jsonify({'success': False, 'message': '请先设置世界背景'}), 400
    
    agents_to_save = [a.copy() for a in state.agents]
    
    if auto_generate and len(agents_to_save) == 0:
        if not state.api_key:
            return jsonify({'success': False, 'message': '自动生成角色需要先配置API Key'}), 400
        
        generated_agents = generate_agents_for_world(state.world, 4)
        if generated_agents:
            agents_to_save = generated_agents
            state.agents = [a.copy() for a in generated_agents]
    
    if not template_id:
        template_id = f"custom_{uuid.uuid4().hex[:8]}"
    
    state.custom_templates[template_id] = {
        'name': template_name,
        'description': data.get('description', '用户自定义模板'),
        'world': state.world.copy(),
        'agents': agents_to_save,
        'custom': True
    }
    
    return jsonify({
        'success': True, 
        'message': f'模板"{template_name}"已保存' + (f'，已生成{len(agents_to_save)}个角色' if auto_generate and agents_to_save else ''),
        'template_id': template_id,
        'agents_generated': len(agents_to_save) if auto_generate else 0
    })

@app.route('/api/templates/update', methods=['POST'])
def update_template():
    data = request.json
    template_id = data.get('id')
    
    if not template_id or template_id not in state.custom_templates:
        return jsonify({'success': False, 'message': '模板不存在'}), 404
    
    template = state.custom_templates[template_id]
    
    if 'name' in data:
        template['name'] = data['name']
    if 'description' in data:
        template['description'] = data['description']
    if 'world' in data:
        template['world'] = data['world']
    if 'agents' in data:
        template['agents'] = data['agents']
    
    return jsonify({'success': True, 'message': '模板已更新'})

@app.route('/api/templates/get/<template_id>')
def get_template(template_id):
    all_templates = {**TEMPLATES, **state.custom_templates}
    if template_id in all_templates:
        return jsonify(all_templates[template_id])
    return jsonify({'error': '模板不存在'}), 404

@app.route('/api/templates/delete', methods=['POST'])
def delete_template():
    template_id = request.json.get('id')
    if template_id in state.custom_templates:
        del state.custom_templates[template_id]
        return jsonify({'success': True, 'message': '模板已删除'})
    return jsonify({'success': False, 'message': '模板不存在'}), 404

@app.route('/api/agents', methods=['GET', 'POST', 'DELETE'])
def agents():
    if request.method == 'POST':
        agent = request.json
        if 'id' not in agent:
            agent['id'] = str(uuid.uuid4())
        
        existing = next((i for i, a in enumerate(state.agents) if a['id'] == agent['id']), None)
        if existing is not None:
            state.agents[existing] = agent
        else:
            state.agents.append(agent)
        
        return jsonify({'success': True, 'agent': agent, 'message': '角色已保存'})
    
    elif request.method == 'DELETE':
        agent_id = request.json.get('id')
        state.agents = [a for a in state.agents if a['id'] != agent_id]
        return jsonify({'success': True, 'message': '角色已删除'})
    
    else:
        return jsonify(state.agents)

@app.route('/api/agents/clear', methods=['POST'])
def clear_agents():
    state.agents = []
    return jsonify({'success': True, 'message': '已清空所有角色'})

@app.route('/api/agents/generate', methods=['POST'])
def generate_agents():
    if not state.api_key:
        return jsonify({'success': False, 'message': '请先配置API Key'}), 400
    
    if not state.world.get('background'):
        return jsonify({'success': False, 'message': '请先设置世界背景'}), 400
    
    count = request.json.get('count', 4)
    
    try:
        agents = generate_agents_for_world(state.world, count)
        if agents:
            # 替换当前角色
            state.agents = agents
            return jsonify({
                'success': True, 
                'message': f'已生成{len(agents)}个角色',
                'agents': agents
            })
        else:
            return jsonify({'success': False, 'message': '生成失败，请重试'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'生成失败: {str(e)}'}), 500

@app.route('/api/metrics', methods=['GET', 'POST', 'DELETE'])
def metrics():
    if request.method == 'POST':
        metric = request.json
        if 'id' not in metric:
            metric['id'] = str(uuid.uuid4())
        
        existing = next((i for i, m in enumerate(state.metrics) if m['id'] == metric['id']), None)
        if existing is not None:
            state.metrics[existing] = metric
        else:
            state.metrics.append(metric)
            state.metric_data[metric['id']] = []
        
        return jsonify({'success': True, 'metric': metric, 'message': '指标已保存'})
    
    elif request.method == 'DELETE':
        metric_id = request.json.get('id')
        state.metrics = [m for m in state.metrics if m['id'] != metric_id]
        if metric_id in state.metric_data:
            del state.metric_data[metric_id]
        return jsonify({'success': True, 'message': '指标已删除'})
    
    else:
        return jsonify(state.metrics)

@app.route('/api/metrics/data')
def get_metric_data():
    return jsonify(state.metric_data)

@app.route('/api/metrics/generate', methods=['POST'])
def generate_metric():
    description = request.json.get('description', '')
    
    if not description:
        return jsonify({'success': False, 'message': '请描述你想观察的指标'}), 400
    
    if not state.api_key:
        return jsonify({'success': False, 'message': '请先配置API Key'}), 400
    
    try:
        prompt = f"""用户想在社会模拟实验中观察一个指标，描述如下：
"{description}"

请帮助生成这个指标的配置。直接返回JSON格式，不要有其他内容：
{{
    "name": "指标名称（简短，2-6个字）",
    "description": "指标的详细描述（说明这个指标衡量什么）",
    "min": 最小值（数字），
    "max": 最大值（数字），
    "unit": "单位（如%、分、个等，可以为空字符串）"
}}"""
        
        messages = [{"role": "user", "content": prompt}]
        response = call_qwen_api(messages, temperature=0.3)
        
        json_match = re.search(r'\{[^{}]+\}', response)
        if json_match:
            metric_config = json.loads(json_match.group())
            metric_config['id'] = str(uuid.uuid4())
            return jsonify({'success': True, 'metric': metric_config})
        else:
            return jsonify({'success': False, 'message': '生成失败，请重试'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': f'生成失败: {str(e)}'}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    if state.running:
        return jsonify({'error': '模拟已在运行中'}), 400
    
    if not state.agents:
        return jsonify({'error': '请先添加角色'}), 400
    
    if not state.api_key:
        return jsonify({'error': '请先设置API Key'}), 400
    
    data = request.json or {}
    state.speed = data.get('speed', 3)
    state.running = True
    
    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()
    
    return jsonify({'success': True})

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    state.running = False
    return jsonify({'success': True})

@app.route('/api/simulation/step', methods=['POST'])
def step_simulation():
    if state.running:
        return jsonify({'error': '请先暂停自动模拟'}), 400
    
    result = run_simulation_step()
    return jsonify({'success': True, 'result': result})

@app.route('/api/simulation/status')
def simulation_status():
    return jsonify({
        'running': state.running,
        'round': state.round,
        'speed': state.speed,
        'agent_count': len(state.agents)
    })

@app.route('/api/history')
def get_history():
    since = request.args.get('since', 0, type=int)
    return jsonify(state.history[since:])

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    state.history = []
    state.round = 0
    state.metric_data = {m['id']: [] for m in state.metrics}
    return jsonify({'success': True})

@app.route('/api/event', methods=['POST'])
def inject_event():
    event = request.json.get('event', '')
    if event:
        state.event_queue.put(event)
        return jsonify({'success': True})
    return jsonify({'error': '事件内容不能为空'}), 400

@app.route('/api/export')
def export_data():
    data = {
        'world': state.world,
        'agents': state.agents,
        'history': state.history,
        'metrics': state.metrics,
        'metric_data': state.metric_data,
        'custom_templates': state.custom_templates,
        'exported_at': datetime.now().isoformat()
    }
    return jsonify(data)

@app.route('/api/import', methods=['POST'])
def import_data():
    data = request.json
    if 'world' in data:
        state.world = data['world']
    if 'agents' in data:
        state.agents = data['agents']
    if 'history' in data:
        state.history = data['history']
        state.round = len(data['history'])
    if 'metrics' in data:
        state.metrics = data['metrics']
    if 'metric_data' in data:
        state.metric_data = data['metric_data']
    if 'custom_templates' in data:
        state.custom_templates = data['custom_templates']
    return jsonify({'success': True, 'message': '数据已导入'})

# ============================================
# 预设模板
# ============================================
TEMPLATES = {
    'ancient_town': {
        'name': '青石古镇',
        'description': '一个传统的中国古镇，居民们各司其职',
        'world': {
            'name': '青石镇',
            'background': '''这是一个位于江南水乡的古老小镇，历史悠久，民风淳朴。
镇上有热闹的集市、古老的茶馆、庄严的私塾，还有各种手工作坊。
人们日出而作，日落而息，遵循着传统的礼俗生活。
最近，镇上来了一些外地商人，带来了新的商品和观念，也带来了一些变化。''',
            'rules': '''1. 社会阶层分明：士农工商，各有地位
2. 经济以银两和铜钱为货币，也可以物易物
3. 邻里关系紧密，消息传播很快
4. 尊老爱幼，重视家族荣誉
5. 镇上有里正维持秩序，重大事务需要商议决定''',
            'resources': '银两、铜钱、粮食、布匹、茶叶、瓷器、声望、人情'
        },
        'agents': [
            {
                'id': 'wang',
                'name': '王掌柜',
                'personality': '精明能干的中年商人，为人圆滑但讲究诚信。善于察言观色，懂得人情世故。对利润敏感，但也重视长期合作关系。偶尔会显得有些市侩，但内心也有柔软的一面。',
                'goal': '扩大自己的绸缎生意，成为镇上最大的商铺，同时为儿子谋一门好亲事',
                'memory': '三十年前从省城来到此地，白手起家打拼出现在的基业。妻子早逝，独自拉扯大儿子。去年刚还清了所有债务，现在手头宽裕了一些。'
            },
            {
                'id': 'li',
                'name': '李老农',
                'personality': '朴实憨厚的老农民，勤劳节俭，性格有些固执保守。不善言辞，但心地善良。对土地有着深厚的感情，对新事物持谨慎态度。',
                'goal': '攒够钱为小儿子娶媳妇，希望家里能添个孙子',
                'memory': '种了一辈子地，对节气变化和天气很敏感。大儿子去年去了城里做工，很少回来。老伴身体不太好，经常需要抓药。'
            },
            {
                'id': 'chen',
                'name': '陈秀才',
                'personality': '饱读诗书的年轻书生，有些理想主义和书生意气。清高自傲，不屑与市井小民为伍，但内心渴望被认可。说话文绉绉的，偶尔会引经据典。',
                'goal': '考取功名，光耀门楣，将来做一个为民请命的好官',
                'memory': '出身寒门，父母省吃俭用供他读书。已经参加过两次乡试，都名落孙山。最近在私塾教书维持生计，心中郁郁不得志。'
            },
            {
                'id': 'zhang',
                'name': '张铁匠',
                'personality': '性格直爽的中年汉子，说话大嗓门，脾气有点急躁但心肠热。手艺精湛，对自己的作品很自豪。重义气，最看不惯欺负人的事。',
                'goal': '把手艺传给儿子，让张家铁铺世世代代传下去',
                'memory': '祖传三代的铁匠手艺，在镇上口碑很好。前年妻子给他生了个儿子，现在天天盼着儿子快点长大学手艺。和王掌柜是老相识，经常一起喝酒。'
            }
        ]
    },
    
    'startup': {
        'name': '创业公司',
        'description': '一家正在快速成长的AI创业公司',
        'world': {
            'name': '星火科技',
            'background': '''星火科技是一家成立两年的AI创业公司，专注于开发智能客服解决方案。
公司位于科技园区的一栋写字楼里，团队共有15人。
目前正处于A轮融资的关键时期，产品刚刚完成2.0版本的开发。
竞争对手最近获得了大额融资，市场压力越来越大。
团队成员来自不同背景，有着不同的理念和工作风格。''',
            'rules': '''1. 公司采用扁平化管理，鼓励开放讨论
2. 每周一有全员例会，周五有周报
3. 资源有限，需要合理分配人力和预算
4. 决策需要核心团队达成共识
5. 用户反馈和数据驱动产品方向''',
            'resources': '资金（runway还剩8个月）、技术人才、客户资源、投资人关系、团队士气'
        },
        'agents': [
            {
                'id': 'alex',
                'name': 'Alex（CEO）',
                'personality': '有远见的领导者，善于激励团队和对外沟通。乐观自信，有时候过于乐观。擅长画大饼，但也确实有执行力。压力大时会焦虑，但不会在团队面前表现出来。',
                'goal': '带领公司成功完成A轮融资，在年底前实现盈亏平衡',
                'memory': '之前在大厂做到了总监，三年前出来创业。第一次创业失败了，这次更加谨慎。上个月刚和投资人吃了饭，对方表示感兴趣但还在观望。'
            },
            {
                'id': 'lisa',
                'name': 'Lisa（CTO）',
                'personality': '技术大牛，追求完美的工程师思维。不善于沟通，说话直接有时会得罪人。对技术债务很敏感，经常和产品需求产生冲突。内心其实很在意团队，但不会表达。',
                'goal': '打造一个技术架构优雅的产品，不想为了赶进度牺牲代码质量',
                'memory': '在大厂工作了五年，技术能力很强。被Alex挖过来的时候承诺给期权。最近连续加班了两周，身体有点吃不消。'
            },
            {
                'id': 'mike',
                'name': 'Mike（销售总监）',
                'personality': '外向健谈，善于社交。结果导向，有时候会过度承诺客户。和技术团队经常有摩擦，觉得他们不理解市场需求。压力大时会变得急躁。',
                'goal': '本季度完成300万的销售目标，拿到大客户订单',
                'memory': '半年前从竞争对手那里跳槽过来，带来了一些客户资源。最近在谈的大客户迟迟不签约，有点焦虑。'
            },
            {
                'id': 'emma',
                'name': 'Emma（产品经理）',
                'personality': '用户思维强，善于平衡各方需求。沟通能力强，经常在技术和销售之间调解。有时候会过于在意他人感受，不敢做决断。内心戏比较多。',
                'goal': '推动产品3.0版本上线，提升用户满意度',
                'memory': '工作三年的产品经理，这是她第一次加入创业公司。最近用户调研发现了一些产品问题，正在考虑如何优先排期。'
            }
        ]
    },
    
    'social_network': {
        'name': '网络社区',
        'description': '模拟信息在社交网络中的传播',
        'world': {
            'name': '微言广场',
            'background': '''微言广场是一个虚拟的社交媒体平台，日活用户数百万。
平台上有各种话题的讨论区，用户可以发帖、评论、点赞、转发。
最近平台上出现了一个热门话题：关于AI是否会取代人类工作的讨论。
不同立场的用户展开了激烈的辩论，氛围有些紧张。
平台算法会优先推荐互动量高的内容。''',
            'rules': '''1. 每个用户有自己的粉丝量和影响力
2. 发帖可能被推荐到热门，也可能石沉大海
3. 情绪化的内容容易获得传播
4. 被大V转发可以快速获得曝光
5. 过激言论可能被举报或限流''',
            'resources': '粉丝数、帖子热度、社交货币、声誉、情绪能量'
        },
        'agents': [
            {
                'id': 'rational',
                'name': '理性派小王',
                'personality': '注重逻辑和事实，喜欢引用数据和研究。说话有条理，但有时候会显得居高临下。对情绪化的讨论很反感，会试图用理性分析来回应。',
                'goal': '传播理性思考的重要性，让更多人学会独立思考',
                'memory': '互联网从业者，研究生学历。去年因为一篇深度分析文章获得了10万粉丝。最近的AI话题让他很感兴趣，已经发了几篇长文。'
            },
            {
                'id': 'emotional',
                'name': '感性派小李',
                'personality': '情感丰富，善于讲故事，容易共情。发言往往能引起很多共鸣，但有时候会过于情绪化。对负面评论很敏感，会感到受伤。',
                'goal': '建立一个温暖的社群，让大家感受到被理解',
                'memory': '自由职业者，曾经历过网络暴力，现在更加谨慎地表达观点。粉丝多是和她有类似经历的人。最近对AI话题有些担忧。'
            },
            {
                'id': 'neutral',
                'name': '中立派老张',
                'personality': '保持中立，喜欢看各方观点。说话温和，善于总结不同意见。有时候会因为太中立而被各方批评为骑墙派。',
                'goal': '促进不同立场的对话，减少网络戾气',
                'memory': '做过多年新闻工作，习惯了客观报道的思维。粉丝量不多但很稳定。最近一直在关注AI话题的讨论，觉得双方都有道理。'
            },
            {
                'id': 'troll',
                'name': '键盘侠小黑',
                'personality': '喜欢抬杠和引战，说话刻薄尖酸。以惹怒他人为乐，但内心其实很孤独。有时候会突然说出一些有见地的话。',
                'goal': '找点乐子，顺便涨涨粉',
                'memory': '匿名账号，无人知道真实身份。之前因为言论过激被封过几个号。最近发现AI话题是引战的好素材。'
            }
        ]
    },
    
    'survival': {
        'name': '末日求生',
        'description': '资源匮乏的末日世界，考验人性',
        'world': {
            'name': '废墟避难所',
            'background': '''大灾变发生已经三年了，城市变成了废墟，幸存者们艰难求生。
这是一个由十几个人组成的避难所，位于一座废弃工厂的地下室。
最近的物资搜索发现，周围几个街区的资源已经被搜刮得差不多了。
远处似乎有其他幸存者团体的活动迹象，不知道是敌是友。
冬天快要来了，必须储备足够的食物和燃料。''',
            'rules': '''1. 资源有限，每天需要消耗食物和水
2. 外出搜索有风险，可能遇到危险
3. 集体决策，但实际上谁有枪谁说话分量更重
4. 新来的人需要经过考验才能被接纳
5. 生病了只能靠自己扛，药物很珍贵''',
            'resources': '食物（剩余15天）、饮用水、药品、武器弹药、燃料、避难所安全'
        },
        'agents': [
            {
                'id': 'leader',
                'name': '老陈（领导者）',
                'personality': '前军人，有领导力和生存技能。做事果断但有时候独断。对团队成员有责任感，但也会在必要时做出残酷的决定。内心深处对灾难前的生活很怀念。',
                'goal': '带领团队活过这个冬天，找到一个更安全的长期据点',
                'memory': '灾变前是退伍兵，有家人但都失散了。建立这个避难所的时候只有5个人，现在发展到十几人。上周做了一个艰难的决定，把一个受伤的人留在了外面。'
            },
            {
                'id': 'doctor',
                'name': '林医生',
                'personality': '灾变前是外科医生，有珍贵的医疗技能。性格温和，坚持医者仁心的原则。有时候和老陈在决策上有冲突，认为不能放弃任何一个人。',
                'goal': '保护好每一个团队成员，找到更多的医疗物资',
                'memory': '医院被丧失理智的人群冲击时逃出来的。医疗包里的药品越来越少，上次有人受伤不得不用最后的抗生素。'
            },
            {
                'id': 'scavenger',
                'name': '小刘（搜索者）',
                'personality': '年轻机灵，擅长在废墟中寻找物资。有点鲁莽，但运气一直不错。对老陈既敬畏又有些不满，觉得自己的贡献没被认可。',
                'goal': '证明自己的价值，获得更多的话语权',
                'memory': '灾变时还是大学生，失去了所有家人。两个月前被这个团队收留，负责外出搜索物资。上次出去发现了另一个团体的痕迹。'
            },
            {
                'id': 'newcomer',
                'name': '阿May（新人）',
                'personality': '刚加入团队一周的年轻女性，沉默寡言。看起来有些心事，不太愿意提自己的过去。有不错的动手能力，修好了避难所的发电机。',
                'goal': '融入这个团队，找到一个可以安心的地方',
                'memory': '之前的团队发生了内讧，只有她一个人逃出来。背包里藏着一些物资没有告诉其他人。不确定是否能信任这里的人。'
            }
        ]
    }
}

# ============================================
# HTML模板
# ============================================
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SocialSim - AI社会模拟平台</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0f;
            --bg-secondary: #12121a;
            --bg-tertiary: #1a1a25;
            --accent: #64ffda;
            --accent-dim: rgba(100, 255, 218, 0.1);
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --text-muted: #555;
            --border: rgba(255, 255, 255, 0.08);
            --danger: #ff6b6b;
            --warning: #ffd93d;
            --success: #6bcb77;
            --purple: #a78bfa;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Noto Sans SC', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        /* Toast */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .toast {
            padding: 12px 20px;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            animation: slideIn 0.3s ease, fadeOut 0.3s ease 2.7s;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        .toast.success { background: linear-gradient(135deg, #6bcb77, #4ade80); }
        .toast.error { background: linear-gradient(135deg, #ff6b6b, #ef4444); }
        .toast.info { background: linear-gradient(135deg, #64ffda, #22d3ee); color: #0a0a0f; }
        
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        
        @keyframes fadeOut {
            from { opacity: 1; }
            to { opacity: 0; }
        }
        
        /* 首页 */
        .home-container {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
            background: radial-gradient(ellipse at center, #1a1a2e 0%, #0a0a0f 70%);
        }
        
        .logo { font-size: 4rem; margin-bottom: 1rem; text-shadow: 0 0 40px rgba(100, 255, 218, 0.5); }
        
        .title {
            font-size: 3rem;
            font-weight: 200;
            letter-spacing: 0.3em;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle { font-size: 1rem; letter-spacing: 0.5em; color: var(--text-secondary); margin-bottom: 2rem; }
        .description { font-size: 1.1rem; color: var(--text-secondary); max-width: 500px; margin-bottom: 2.5rem; line-height: 1.8; }
        
        .primary-btn {
            background: linear-gradient(135deg, var(--accent), #4fd1c7);
            color: var(--bg-primary);
            border: none;
            padding: 1rem 2.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 20px rgba(100, 255, 218, 0.3);
        }
        
        .primary-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 30px rgba(100, 255, 218, 0.4); }
        
        .features { display: flex; gap: 2rem; margin-top: 3rem; flex-wrap: wrap; justify-content: center; }
        .feature { display: flex; align-items: center; gap: 0.5rem; color: var(--text-secondary); font-size: 0.9rem; }
        
        /* 主应用 */
        .app-container { display: none; min-height: 100vh; }
        
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.75rem 1.5rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .header-left { display: flex; align-items: center; gap: 1rem; }
        .header-logo { font-size: 1.5rem; color: var(--accent); }
        .header-title { font-size: 1rem; font-weight: 500; }
        .header-right { display: flex; align-items: center; gap: 0.75rem; }
        
        .btn {
            padding: 0.5rem 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text-primary);
            cursor: pointer;
            font-size: 0.875rem;
            transition: all 0.2s;
        }
        
        .btn:hover { background: var(--bg-tertiary); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-accent { background: var(--accent); color: var(--bg-primary); border-color: var(--accent); font-weight: 600; }
        .btn-accent:hover { background: #4fd1c7; }
        .btn-danger { border-color: var(--danger); color: var(--danger); }
        .btn-danger:hover { background: rgba(255, 107, 107, 0.1); }
        .btn-purple { border-color: var(--purple); color: var(--purple); }
        .btn-purple:hover { background: rgba(167, 139, 250, 0.1); }
        .btn-sm { padding: 0.375rem 0.75rem; font-size: 0.8rem; }
        
        /* 标签页 */
        .tabs {
            display: flex;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            overflow-x: auto;
        }
        
        .tab {
            padding: 0.875rem 1.5rem;
            cursor: pointer;
            color: var(--text-secondary);
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
            font-size: 0.9rem;
            white-space: nowrap;
        }
        
        .tab:hover { color: var(--text-primary); background: var(--bg-tertiary); }
        .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
        
        /* 内容区域 */
        .content { min-height: calc(100vh - 110px); }
        .panel { display: none; width: 100%; padding: 1.5rem; }
        .panel.active { display: block; }
        
        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .panel-title { font-size: 1.25rem; font-weight: 500; }
        .panel-actions { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        
        /* 表单 */
        .form-group { margin-bottom: 1.25rem; }
        .form-label { display: block; font-size: 0.875rem; color: var(--accent); margin-bottom: 0.5rem; font-weight: 500; }
        
        .form-input, .form-textarea, .form-select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text-primary);
            font-size: 0.95rem;
            font-family: inherit;
            transition: border-color 0.2s;
        }
        
        .form-input:focus, .form-textarea:focus, .form-select:focus { outline: none; border-color: var(--accent); }
        .form-textarea { resize: vertical; min-height: 100px; line-height: 1.6; }
        .form-hint { font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        
        /* 卡片 */
        .card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }
        
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .card-title { font-size: 1rem; font-weight: 500; }
        
        /* 模板网格 */
        .template-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .template-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }
        
        .template-card:hover { border-color: var(--accent); transform: translateY(-2px); }
        .template-card.custom { border-style: dashed; border-color: var(--purple); }
        .template-icon { font-size: 2rem; margin-bottom: 0.75rem; }
        .template-name { font-size: 1rem; font-weight: 500; margin-bottom: 0.375rem; }
        .template-desc { font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem; }
        
        .template-actions {
            display: none;
            position: absolute;
            top: 8px;
            right: 8px;
            gap: 4px;
        }
        
        .template-card.custom:hover .template-actions { display: flex; }
        
        .template-action-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            width: 28px;
            height: 28px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .template-action-btn:hover { background: var(--bg-primary); color: var(--text-primary); }
        .template-action-btn.danger:hover { color: var(--danger); border-color: var(--danger); }
        
        /* 角色编辑器 */
        .editor-layout {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1.5rem;
            max-width: 1200px;
            align-items: start;
        }
        
        .editor-sidebar {
            position: sticky;
            top: 130px;
            max-height: calc(100vh - 150px);
            overflow-y: auto;
        }
        
        .agent-list { display: flex; flex-direction: column; gap: 0.75rem; }
        
        .agent-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .agent-card:hover { border-color: rgba(100, 255, 218, 0.3); }
        .agent-card.active { border-color: var(--accent); background: var(--accent-dim); }
        .agent-header { display: flex; align-items: center; gap: 0.75rem; }
        
        .agent-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-primary);
            font-weight: 600;
            font-size: 1rem;
            flex-shrink: 0;
        }
        
        .agent-info { flex: 1; min-width: 0; }
        .agent-name { font-weight: 500; margin-bottom: 0.25rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .agent-preview { font-size: 0.8rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        
        .agent-editor {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        .editor-title {
            font-size: 1rem;
            font-weight: 500;
            color: var(--accent);
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }
        
        .editor-actions { display: flex; gap: 0.75rem; margin-top: 1.5rem; flex-wrap: wrap; }
        
        /* 模拟面板 */
        .sim-container {
            display: grid;
            grid-template-columns: 260px 1fr 300px;
            height: calc(100vh - 110px);
            gap: 1px;
            background: var(--border);
        }
        
        .sim-sidebar {
            background: var(--bg-primary);
            padding: 1.25rem;
            overflow-y: auto;
        }
        
        .sim-main {
            background: var(--bg-primary);
            display: flex;
            flex-direction: column;
            min-height: 0;
        }
        
        .sim-sidebar-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 1rem;
        }
        
        .sim-agent {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem;
            background: var(--bg-secondary);
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        
        .sim-agent-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent), #a78bfa);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--bg-primary);
            font-weight: 600;
            font-size: 0.875rem;
            flex-shrink: 0;
        }
        
        .sim-agent-name { font-size: 0.9rem; font-weight: 500; }
        .sim-agent-goal { font-size: 0.75rem; color: var(--text-secondary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        
        .sim-controls {
            padding: 1rem 1.25rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            flex-shrink: 0;
        }
        
        .control-group { display: flex; align-items: center; gap: 0.5rem; }
        .round-badge { background: var(--bg-tertiary); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; color: var(--text-secondary); }
        .status-badge { padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
        .status-running { background: rgba(107, 203, 119, 0.2); color: var(--success); }
        .status-stopped { background: rgba(255, 217, 61, 0.2); color: var(--warning); }
        
        .sim-logs { flex: 1; overflow-y: auto; padding: 1rem; min-height: 0; }
        
        .log-entry {
            margin-bottom: 1rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 10px;
            border-left: 3px solid var(--accent);
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .log-entry.error { border-left-color: var(--danger); }
        .log-entry.event { border-left-color: var(--warning); background: rgba(255, 217, 61, 0.05); }
        .log-meta { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; font-size: 0.8rem; flex-wrap: wrap; }
        .log-round { color: var(--accent); font-weight: 600; }
        .log-agent { color: #a78bfa; font-weight: 500; }
        .log-time { color: var(--text-muted); margin-left: auto; }
        .log-content { color: var(--text-primary); line-height: 1.7; font-size: 0.95rem; white-space: pre-wrap; word-break: break-word; }
        .log-event-tag { display: inline-block; background: rgba(255, 217, 61, 0.2); color: var(--warning); padding: 0.125rem 0.5rem; border-radius: 4px; font-size: 0.75rem; margin-bottom: 0.5rem; }
        .empty-logs { text-align: center; color: var(--text-muted); padding: 3rem; }
        
        .event-input-container { padding: 1rem 1.25rem; background: var(--bg-secondary); border-top: 1px solid var(--border); flex-shrink: 0; }
        .event-input-wrapper { display: flex; gap: 0.5rem; }
        .event-input { flex: 1; padding: 0.75rem 1rem; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); font-size: 0.9rem; }
        .event-input:focus { outline: none; border-color: var(--accent); }
        
        .world-section { margin-bottom: 1.25rem; }
        .world-label { font-size: 0.7rem; color: var(--accent); font-weight: 600; margin-bottom: 0.375rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .world-text { font-size: 0.85rem; color: var(--text-secondary); line-height: 1.6; white-space: pre-wrap; word-break: break-word; max-height: 150px; overflow-y: auto; }
        
        .metrics-panel { margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--border); }
        .metric-item { background: var(--bg-secondary); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.5rem; }
        .metric-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
        .metric-name { font-size: 0.85rem; font-weight: 500; }
        .metric-value { font-size: 0.9rem; color: var(--accent); font-weight: 600; }
        .metric-bar { height: 4px; background: var(--bg-tertiary); border-radius: 2px; overflow: hidden; }
        .metric-bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #a78bfa); transition: width 0.5s ease; }
        
        .chart-container { background: var(--bg-secondary); border-radius: 12px; padding: 1rem; margin-top: 1rem; height: 250px; }
        
        .metric-config-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .metric-config-info { flex: 1; }
        .metric-config-name { font-weight: 500; margin-bottom: 0.25rem; }
        .metric-config-desc { font-size: 0.8rem; color: var(--text-secondary); }
        .metric-config-range { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }
        
        .ai-input-group { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .ai-input { flex: 1; padding: 0.75rem 1rem; background: var(--bg-tertiary); border: 1px solid var(--border); border-radius: 8px; color: var(--text-primary); font-size: 0.9rem; }
        .ai-input:focus { outline: none; border-color: var(--accent); }
        
        .settings-grid { display: grid; gap: 1.5rem; max-width: 800px; }
        .api-status { display: flex; align-items: center; gap: 0.5rem; font-size: 0.875rem; }
        .api-status-dot { width: 8px; height: 8px; border-radius: 50%; }
        .api-status-dot.connected { background: var(--success); }
        .api-status-dot.disconnected { background: var(--danger); }
        
        .modal-overlay {
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            visibility: hidden;
            transition: all 0.2s;
        }
        
        .modal-overlay.active { opacity: 1; visibility: visible; }
        
        .modal {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            max-width: 500px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.25rem; }
        .modal-title { font-size: 1.1rem; font-weight: 500; }
        .modal-close { background: none; border: none; color: var(--text-secondary); font-size: 1.5rem; cursor: pointer; line-height: 1; }
        .modal-footer { display: flex; justify-content: flex-end; gap: 0.75rem; margin-top: 1.5rem; }
        
        .speed-control { margin-top: 1.5rem; padding: 1rem; background: var(--bg-secondary); border-radius: 10px; }
        .speed-slider { width: 100%; margin: 0.5rem 0; -webkit-appearance: none; background: var(--bg-tertiary); height: 6px; border-radius: 3px; }
        .speed-slider::-webkit-slider-thumb { -webkit-appearance: none; width: 16px; height: 16px; background: var(--accent); border-radius: 50%; cursor: pointer; }
        
        .checkbox-group { display: flex; align-items: center; gap: 0.5rem; margin-top: 1rem; }
        .checkbox-group input { width: 18px; height: 18px; accent-color: var(--accent); }
        .checkbox-group label { font-size: 0.9rem; color: var(--text-secondary); }
        
        .generate-hint {
            background: rgba(100, 255, 218, 0.1);
            border: 1px solid rgba(100, 255, 218, 0.2);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .generate-hint strong { color: var(--accent); }
        
        @media (max-width: 1024px) {
            .sim-container { grid-template-columns: 1fr; height: auto; min-height: calc(100vh - 110px); }
            .sim-sidebar { display: none; }
            .sim-main { height: calc(100vh - 110px); }
            .editor-layout { grid-template-columns: 1fr; }
            .editor-sidebar { position: static; max-height: none; }
        }
        
        @media (max-width: 640px) {
            .panel { padding: 1rem; }
            .form-row { grid-template-columns: 1fr; }
            .panel-header { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="toast-container" id="toast-container"></div>
    
    <!-- 首页 -->
    <div class="home-container" id="home">
        <div class="logo">◈</div>
        <h1 class="title">SocialSim</h1>
        <p class="subtitle">AI 社会模拟沙盒</p>
        <p class="description">
            创建虚拟世界，添加 AI 角色，观察社会动态的涌现。<br>
            从经济博弈到文化演化，探索无限可能。
        </p>
        <button class="primary-btn" onclick="enterApp()">开始创建实验 →</button>
        <div class="features">
            <div class="feature">🌍 自由定义世界规则</div>
            <div class="feature">🤖 AI自动生成角色</div>
            <div class="feature">📊 自定义观察指标</div>
            <div class="feature">⚡ 随时注入事件干预</div>
        </div>
    </div>
    
    <!-- 主应用 -->
    <div class="app-container" id="app">
        <header class="header">
            <div class="header-left">
                <span class="header-logo">◈</span>
                <span class="header-title">SocialSim</span>
            </div>
            <div class="header-right">
                <button class="btn btn-sm" onclick="exportData()">📤 导出</button>
                <button class="btn btn-sm" onclick="showImportModal()">📥 导入</button>
                <button class="btn btn-sm" onclick="goHome()">🏠 首页</button>
            </div>
        </header>
        
        <div class="tabs">
            <div class="tab active" data-tab="world" onclick="switchTab('world')">🌍 世界设定</div>
            <div class="tab" data-tab="agents" onclick="switchTab('agents')">👥 角色管理</div>
            <div class="tab" data-tab="metrics" onclick="switchTab('metrics')">📊 观察指标</div>
            <div class="tab" data-tab="simulate" onclick="switchTab('simulate')">▶️ 模拟运行</div>
            <div class="tab" data-tab="settings" onclick="switchTab('settings')">⚙️ 设置</div>
        </div>
        
        <div class="content">
            <!-- 世界设定 -->
            <div class="panel active" id="panel-world">
                <div class="panel-header">
                    <h2 class="panel-title">选择模板或自定义世界</h2>
                    <div class="panel-actions">
                        <button class="btn btn-purple" onclick="showSaveTemplateModal()">💾 保存为模板</button>
                    </div>
                </div>
                
                <div class="template-grid" id="template-grid"></div>
                
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">世界设定</h3>
                    </div>
                    <div class="form-group">
                        <label class="form-label">世界名称</label>
                        <input type="text" class="form-input" id="world-name" placeholder="给你的世界起个名字...">
                    </div>
                    <div class="form-group">
                        <label class="form-label">背景描述</label>
                        <textarea class="form-textarea" id="world-background" rows="5" 
                            placeholder="描述这个世界的时代背景、地理环境、社会结构..."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">运行规则</label>
                        <textarea class="form-textarea" id="world-rules" rows="5" 
                            placeholder="定义这个世界运行的规则..."></textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label">资源要素</label>
                        <input type="text" class="form-input" id="world-resources" 
                            placeholder="金币、粮食、声望、魔法水晶...">
                    </div>
                    <button class="btn btn-accent" onclick="saveWorld()">💾 保存世界设定</button>
                </div>
            </div>
            
            <!-- 角色管理 -->
            <div class="panel" id="panel-agents">
                <div class="panel-header">
                    <h2 class="panel-title">角色管理</h2>
                    <div class="panel-actions">
                        <button class="btn btn-purple" onclick="generateAgentsAI()" id="generate-agents-btn">🤖 AI生成角色</button>
                        <button class="btn btn-accent" onclick="addAgent()">➕ 手动添加</button>
                    </div>
                </div>
                
                <div class="generate-hint" id="generate-hint">
                    <strong>💡 提示：</strong>设置好世界背景后，点击"AI生成角色"可自动生成4个适合该世界的角色。
                </div>
                
                <div class="editor-layout">
                    <div class="editor-sidebar">
                        <div class="agent-list" id="agent-list"></div>
                    </div>
                    <div class="agent-editor" id="agent-editor" style="display: none;">
                        <h3 class="editor-title">编辑角色</h3>
                        <div class="form-group">
                            <label class="form-label">角色名称</label>
                            <input type="text" class="form-input" id="agent-name" placeholder="角色名称">
                        </div>
                        <div class="form-group">
                            <label class="form-label">性格特征</label>
                            <textarea class="form-textarea" id="agent-personality" rows="4"
                                placeholder="描述这个角色的性格、价值观、行为倾向..."></textarea>
                            <p class="form-hint">越详细越好，这将决定AI如何扮演这个角色</p>
                        </div>
                        <div class="form-group">
                            <label class="form-label">核心目标</label>
                            <textarea class="form-textarea" id="agent-goal" rows="3"
                                placeholder="这个角色想要达成什么目标？是什么驱动着他/她？"></textarea>
                        </div>
                        <div class="form-group">
                            <label class="form-label">背景记忆</label>
                            <textarea class="form-textarea" id="agent-memory" rows="3"
                                placeholder="角色的过往经历、重要记忆、与其他角色的关系..."></textarea>
                        </div>
                        <div class="editor-actions">
                            <button class="btn btn-accent" onclick="saveAgent()">💾 保存角色</button>
                            <button class="btn btn-danger" onclick="deleteAgent()">🗑️ 删除角色</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 观察指标 -->
            <div class="panel" id="panel-metrics">
                <div class="panel-header">
                    <h2 class="panel-title">观察指标配置</h2>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">🤖 AI辅助添加指标</h3>
                    </div>
                    <p style="font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 1rem;">
                        描述你想观察的指标，AI会自动帮你配置
                    </p>
                    <div class="ai-input-group">
                        <input type="text" class="ai-input" id="metric-description" 
                            placeholder="例如：群体的整体幸福感 / 社区的信任度 / 资源分配的公平性...">
                        <button class="btn btn-accent" onclick="generateMetric()" id="generate-metric-btn">✨ 生成</button>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">📊 已配置的指标</h3>
                    </div>
                    <div id="metrics-list"></div>
                    <div id="metrics-empty" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                        还没有添加观察指标，使用上方AI助手添加一个吧
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">➕ 手动添加指标</h3>
                    </div>
                    <div class="form-group">
                        <label class="form-label">指标名称</label>
                        <input type="text" class="form-input" id="manual-metric-name" placeholder="例如：社会信任度">
                    </div>
                    <div class="form-group">
                        <label class="form-label">指标描述</label>
                        <input type="text" class="form-input" id="manual-metric-desc" placeholder="描述这个指标衡量什么">
                    </div>
                    <div class="form-row">
                        <div class="form-group">
                            <label class="form-label">最小值</label>
                            <input type="number" class="form-input" id="manual-metric-min" value="0">
                        </div>
                        <div class="form-group">
                            <label class="form-label">最大值</label>
                            <input type="number" class="form-input" id="manual-metric-max" value="100">
                        </div>
                    </div>
                    <button class="btn btn-accent" onclick="addManualMetric()">➕ 添加指标</button>
                </div>
            </div>
            
            <!-- 模拟运行 -->
            <div class="panel" id="panel-simulate" style="padding: 0;">
                <div class="sim-container">
                    <div class="sim-sidebar">
                        <div class="sim-sidebar-title">角色列表</div>
                        <div id="sim-agent-list"></div>
                        
                        <div class="speed-control">
                            <div class="sim-sidebar-title">模拟速度</div>
                            <input type="range" class="speed-slider" id="speed-slider" min="1" max="10" value="3">
                            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                                <span id="speed-value">3</span> 秒/回合
                            </div>
                        </div>
                        
                        <div class="metrics-panel" id="sim-metrics-panel" style="display: none;">
                            <div class="sim-sidebar-title">观察指标</div>
                            <div id="sim-metrics-list"></div>
                        </div>
                    </div>
                    
                    <div class="sim-main">
                        <div class="sim-controls">
                            <div class="control-group">
                                <span class="round-badge">回合 <span id="round-display">0</span></span>
                                <span class="status-badge status-stopped" id="status-badge">已停止</span>
                            </div>
                            <div class="control-group" style="margin-left: auto;">
                                <button class="btn btn-sm" id="step-btn" onclick="stepSimulation()">⏭️ 单步</button>
                                <button class="btn btn-sm btn-accent" id="start-btn" onclick="toggleSimulation()">▶️ 开始</button>
                                <button class="btn btn-sm" onclick="clearHistory()">🗑️ 清空</button>
                            </div>
                        </div>
                        
                        <div class="sim-logs" id="sim-logs">
                            <div class="empty-logs">点击"开始"或"单步"按钮启动模拟...</div>
                        </div>
                        
                        <div class="event-input-container">
                            <div class="event-input-wrapper">
                                <input type="text" class="event-input" id="event-input" 
                                    placeholder="注入事件... (例如: 突然发生地震，所有人都惊慌失措)">
                                <button class="btn btn-accent" onclick="injectEvent()">⚡ 注入</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="sim-sidebar">
                        <div class="sim-sidebar-title">世界信息</div>
                        <div class="world-section">
                            <div class="world-label">名称</div>
                            <div class="world-text" id="sim-world-name">-</div>
                        </div>
                        <div class="world-section">
                            <div class="world-label">背景</div>
                            <div class="world-text" id="sim-world-background">-</div>
                        </div>
                        <div class="world-section">
                            <div class="world-label">规则</div>
                            <div class="world-text" id="sim-world-rules">-</div>
                        </div>
                        
                        <div id="chart-section" style="display: none; margin-top: 1.5rem;">
                            <div class="sim-sidebar-title">指标趋势</div>
                            <div class="chart-container">
                                <canvas id="metrics-chart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 设置 -->
            <div class="panel" id="panel-settings">
                <div class="panel-header">
                    <h2 class="panel-title">设置</h2>
                </div>
                
                <div class="settings-grid">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">API 配置</h3>
                            <div class="api-status">
                                <span class="api-status-dot" id="api-status-dot"></span>
                                <span id="api-status-text">未连接</span>
                            </div>
                        </div>
                        <div class="form-group">
                            <label class="form-label">Qwen API Key</label>
                            <input type="password" class="form-input" id="api-key" placeholder="sk-xxxxxxxxxxxxxxxx">
                            <p class="form-hint">从阿里云DashScope获取API Key</p>
                        </div>
                        <div class="form-group">
                            <label class="form-label">模型选择</label>
                            <select class="form-input form-select" id="model-select">
                                <option value="qwen-plus">Qwen-Plus (推荐)</option>
                                <option value="qwen-turbo">Qwen-Turbo (更快)</option>
                                <option value="qwen-max">Qwen-Max (最强)</option>
                            </select>
                        </div>
                        <button class="btn btn-accent" onclick="saveConfig()">💾 保存配置</button>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">使用说明</h3>
                        </div>
                        <div style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.8;">
                            <p><strong>2. 创建世界：</strong>选择模板或自定义，设置后可保存为新模板</p>
                            <p><strong>3. 添加角色：</strong>使用AI自动生成或手动添加</p>
                            <p><strong>4. 配置指标：</strong>使用AI助手快速添加观察指标</p>
                            <p><strong>5. 开始模拟：</strong>观察AI角色互动，实时查看指标变化</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 保存模板模态框 -->
    <div class="modal-overlay" id="save-template-modal">
        <div class="modal">
            <div class="modal-header">
                <h3 class="modal-title" id="template-modal-title">保存为模板</h3>
                <button class="modal-close" onclick="hideSaveTemplateModal()">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">模板名称</label>
                <input type="text" class="form-input" id="template-name" placeholder="给模板起个名字...">
            </div>
            <div class="form-group">
                <label class="form-label">模板描述</label>
                <input type="text" class="form-input" id="template-desc" placeholder="简短描述这个模板...">
            </div>
            <div class="checkbox-group" id="auto-generate-group">
                <input type="checkbox" id="auto-generate-agents" checked>
                <label for="auto-generate-agents">如果没有角色，AI自动生成匹配的角色</label>
            </div>
            <input type="hidden" id="editing-template-id">
            <div class="modal-footer">
                <button class="btn" onclick="hideSaveTemplateModal()">取消</button>
                <button class="btn btn-accent" onclick="saveAsTemplate()">保存</button>
            </div>
        </div>
    </div>
    
    <!-- 导入模态框 -->
    <div class="modal-overlay" id="import-modal">
        <div class="modal">
            <div class="modal-header">
                <h3 class="modal-title">导入实验数据</h3>
                <button class="modal-close" onclick="hideImportModal()">×</button>
            </div>
            <div class="form-group">
                <label class="form-label">粘贴JSON数据</label>
                <textarea class="form-textarea" id="import-data" rows="10" placeholder='{"world": {...}, "agents": [...] }'></textarea>
            </div>
            <div class="modal-footer">
                <button class="btn" onclick="hideImportModal()">取消</button>
                <button class="btn btn-accent" onclick="importData()">导入</button>
            </div>
        </div>
    </div>

    <script>
        // 状态管理
        let state = {
            world: {},
            agents: [],
            metrics: [],
            currentAgentId: null,
            running: false,
            round: 0,
            historyLength: 0
        };
        
        let pollInterval = null;
        let metricsChart = null;
        
        // Toast
        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.textContent = message;
            container.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }
        
        // 页面导航
        function enterApp() {
            document.getElementById('home').style.display = 'none';
            document.getElementById('app').style.display = 'block';
            loadConfig();
            loadTemplates();
            loadAgents();
            loadMetrics();
        }
        
        function goHome() {
            if (state.running) {
                if (!confirm('模拟正在运行，确定要返回首页吗？')) return;
                stopSimulation();
            }
            document.getElementById('home').style.display = 'flex';
            document.getElementById('app').style.display = 'none';
        }
        
        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
            document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
            document.getElementById(`panel-${tabName}`).classList.add('active');
            
            if (tabName === 'simulate') {
                updateSimPanel();
                initMetricsChart();
            }
            if (tabName === 'agents') {
                updateGenerateHint();
            }
        }
        
        // API调用
        async function apiCall(endpoint, method = 'GET', data = null) {
            const options = { method, headers: { 'Content-Type': 'application/json' } };
            if (data) options.body = JSON.stringify(data);
            const response = await fetch(endpoint, options);
            return response.json();
        }
        
        // 配置管理
        async function loadConfig() {
            const config = await apiCall('/api/config');
            updateApiStatus(config.has_key);
            document.getElementById('model-select').value = config.model;
        }
        
        async function saveConfig() {
            const apiKey = document.getElementById('api-key').value;
            const model = document.getElementById('model-select').value;
            await apiCall('/api/config', 'POST', { api_key: apiKey, model });
            updateApiStatus(!!apiKey);
            showToast('配置已保存', 'success');
        }
        
        function updateApiStatus(connected) {
            const dot = document.getElementById('api-status-dot');
            const text = document.getElementById('api-status-text');
            if (connected) {
                dot.classList.add('connected');
                dot.classList.remove('disconnected');
                text.textContent = '已连接';
            } else {
                dot.classList.remove('connected');
                dot.classList.add('disconnected');
                text.textContent = '未连接';
            }
        }
        
        // 模板管理
        async function loadTemplates() {
            const templates = await apiCall('/api/templates');
            const grid = document.getElementById('template-grid');
            
            const icons = { ancient_town: '🏘️', startup: '🏢', social_network: '🌐', survival: '☢️' };
            
            grid.innerHTML = Object.entries(templates).map(([key, t]) => `
                <div class="template-card ${t.custom ? 'custom' : ''}" onclick="applyTemplate('${key}')">
                    ${t.custom ? `
                        <div class="template-actions">
                            <button class="template-action-btn" onclick="event.stopPropagation(); editTemplate('${key}')" title="编辑">✏️</button>
                            <button class="template-action-btn danger" onclick="event.stopPropagation(); deleteTemplate('${key}')" title="删除">×</button>
                        </div>
                    ` : ''}
                    <div class="template-icon">${icons[key] || (t.custom ? '📁' : '📝')}</div>
                    <div class="template-name">${t.name}</div>
                    <div class="template-desc">${t.description}</div>
                    ${t.agents ? `<div style="font-size: 0.75rem; color: var(--accent); margin-top: 0.5rem;">${t.agents.length} 个角色</div>` : ''}
                </div>
            `).join('');
        }
        
        async function applyTemplate(key) {
            const templates = await apiCall('/api/templates');
            const template = templates[key];
            if (!template) return;
            
            // 更新世界设定
            state.world = template.world;
            document.getElementById('world-name').value = template.world.name || '';
            document.getElementById('world-background').value = template.world.background || '';
            document.getElementById('world-rules').value = template.world.rules || '';
            document.getElementById('world-resources').value = template.world.resources || '';
            await saveWorld(false);
            
            // 清空并加载模板角色
            await apiCall('/api/agents/clear', 'POST');
            
            if (template.agents && template.agents.length > 0) {
                for (const agent of template.agents) {
                    // 确保每个角色有新的ID
                    const newAgent = { ...agent, id: undefined };
                    await apiCall('/api/agents', 'POST', newAgent);
                }
            }
            
            await loadAgents();
            showToast(`已应用模板：${template.name}`, 'success');
        }
        
        function showSaveTemplateModal(templateId = null) {
            const modal = document.getElementById('save-template-modal');
            const titleEl = document.getElementById('template-modal-title');
            const autoGenGroup = document.getElementById('auto-generate-group');
            
            if (templateId) {
                // 编辑模式
                titleEl.textContent = '编辑模板';
                autoGenGroup.style.display = 'none';
                document.getElementById('editing-template-id').value = templateId;
                
                // 加载模板数据
                apiCall(`/api/templates/get/${templateId}`).then(template => {
                    document.getElementById('template-name').value = template.name || '';
                    document.getElementById('template-desc').value = template.description || '';
                });
            } else {
                // 新建模式
                titleEl.textContent = '保存为模板';
                autoGenGroup.style.display = 'flex';
                document.getElementById('editing-template-id').value = '';
                document.getElementById('template-name').value = '';
                document.getElementById('template-desc').value = '';
            }
            
            modal.classList.add('active');
        }
        
        function hideSaveTemplateModal() {
            document.getElementById('save-template-modal').classList.remove('active');
        }
        
        async function saveAsTemplate() {
            const name = document.getElementById('template-name').value.trim();
            const desc = document.getElementById('template-desc').value.trim();
            const autoGenerate = document.getElementById('auto-generate-agents').checked;
            const editingId = document.getElementById('editing-template-id').value;
            
            if (!name) {
                showToast('请输入模板名称', 'error');
                return;
            }
            
            const btn = document.querySelector('#save-template-modal .btn-accent');
            btn.disabled = true;
            btn.textContent = '保存中...';
            
            try {
                const result = await apiCall('/api/templates/save', 'POST', {
                    id: editingId || undefined,
                    name,
                    description: desc || '用户自定义模板',
                    auto_generate: autoGenerate && !editingId
                });
                
                if (result.success) {
                    showToast(result.message, 'success');
                    hideSaveTemplateModal();
                    loadTemplates();
                    
                    // 如果生成了角色，重新加载
                    if (result.agents_generated > 0) {
                        await loadAgents();
                    }
                } else {
                    showToast(result.message, 'error');
                }
            } finally {
                btn.disabled = false;
                btn.textContent = '保存';
            }
        }
        
        function editTemplate(id) {
            showSaveTemplateModal(id);
        }
        
        async function deleteTemplate(id) {
            if (!confirm('确定要删除这个模板吗？')) return;
            const result = await apiCall('/api/templates/delete', 'POST', { id });
            if (result.success) {
                showToast('模板已删除', 'success');
                loadTemplates();
            }
        }
        
        // 世界管理
        async function saveWorld(showNotification = true) {
            state.world = {
                name: document.getElementById('world-name').value,
                background: document.getElementById('world-background').value,
                rules: document.getElementById('world-rules').value,
                resources: document.getElementById('world-resources').value
            };
            await apiCall('/api/world', 'POST', state.world);
            if (showNotification) showToast('世界设定已保存', 'success');
            updateGenerateHint();
        }
        
        async function loadWorld() {
            state.world = await apiCall('/api/world');
            document.getElementById('world-name').value = state.world.name || '';
            document.getElementById('world-background').value = state.world.background || '';
            document.getElementById('world-rules').value = state.world.rules || '';
            document.getElementById('world-resources').value = state.world.resources || '';
        }
        
        // 角色管理
        async function loadAgents() {
            state.agents = await apiCall('/api/agents');
            renderAgentList();
            updateGenerateHint();
        }
        
        function updateGenerateHint() {
            const hint = document.getElementById('generate-hint');
            if (state.agents.length > 0) {
                hint.style.display = 'none';
            } else {
                hint.style.display = 'block';
            }
        }
        
        function renderAgentList() {
            const list = document.getElementById('agent-list');
            
            if (state.agents.length === 0) {
                list.innerHTML = '<div style="color: var(--text-muted); text-align: center; padding: 2rem;">还没有角色<br><small>点击"AI生成角色"或"手动添加"</small></div>';
                return;
            }
            
            list.innerHTML = state.agents.map(a => `
                <div class="agent-card ${state.currentAgentId === a.id ? 'active' : ''}" onclick="selectAgent('${a.id}')">
                    <div class="agent-header">
                        <div class="agent-avatar">${(a.name || '?')[0]}</div>
                        <div class="agent-info">
                            <div class="agent-name">${a.name || '未命名'}</div>
                            <div class="agent-preview">${(a.personality || '点击编辑...').slice(0, 30)}...</div>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        function selectAgent(id) {
            state.currentAgentId = id;
            renderAgentList();
            
            const agent = state.agents.find(a => a.id === id);
            if (!agent) return;
            
            document.getElementById('agent-editor').style.display = 'block';
            document.getElementById('agent-name').value = agent.name || '';
            document.getElementById('agent-personality').value = agent.personality || '';
            document.getElementById('agent-goal').value = agent.goal || '';
            document.getElementById('agent-memory').value = agent.memory || '';
        }
        
        async function addAgent() {
            const result = await apiCall('/api/agents', 'POST', {
                name: `角色 ${state.agents.length + 1}`,
                personality: '',
                goal: '',
                memory: ''
            });
            await loadAgents();
            selectAgent(result.agent.id);
            showToast('角色已添加', 'success');
        }
        
        async function saveAgent() {
            if (!state.currentAgentId) return;
            
            const agent = {
                id: state.currentAgentId,
                name: document.getElementById('agent-name').value,
                personality: document.getElementById('agent-personality').value,
                goal: document.getElementById('agent-goal').value,
                memory: document.getElementById('agent-memory').value
            };
            
            await apiCall('/api/agents', 'POST', agent);
            await loadAgents();
            selectAgent(state.currentAgentId);
            showToast('角色已保存', 'success');
        }
        
        async function deleteAgent() {
            if (!state.currentAgentId) return;
            if (!confirm('确定要删除这个角色吗？')) return;
            
            await apiCall('/api/agents', 'DELETE', { id: state.currentAgentId });
            state.currentAgentId = null;
            document.getElementById('agent-editor').style.display = 'none';
            await loadAgents();
            showToast('角色已删除', 'success');
        }
        
        async function generateAgentsAI() {
            if (!state.world.background) {
                showToast('请先设置世界背景', 'error');
                return;
            }
            
            const btn = document.getElementById('generate-agents-btn');
            btn.disabled = true;
            btn.textContent = '🤖 生成中...';
            
            try {
                const result = await apiCall('/api/agents/generate', 'POST', { count: 4 });
                
                if (result.success) {
                    await loadAgents();
                    showToast(result.message, 'success');
                } else {
                    showToast(result.message, 'error');
                }
            } catch (e) {
                showToast('生成失败，请检查API配置', 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '🤖 AI生成角色';
            }
        }
        
        // 指标管理
        async function loadMetrics() {
            state.metrics = await apiCall('/api/metrics');
            renderMetricsList();
        }
        
        function renderMetricsList() {
            const list = document.getElementById('metrics-list');
            const empty = document.getElementById('metrics-empty');
            
            if (state.metrics.length === 0) {
                list.innerHTML = '';
                empty.style.display = 'block';
                return;
            }
            
            empty.style.display = 'none';
            list.innerHTML = state.metrics.map(m => `
                <div class="metric-config-card">
                    <div class="metric-config-info">
                        <div class="metric-config-name">${m.name}</div>
                        <div class="metric-config-desc">${m.description || ''}</div>
                        <div class="metric-config-range">范围: ${m.min || 0} - ${m.max || 100} ${m.unit || ''}</div>
                    </div>
                    <button class="btn btn-sm btn-danger" onclick="deleteMetric('${m.id}')">删除</button>
                </div>
            `).join('');
        }
        
        async function generateMetric() {
            const desc = document.getElementById('metric-description').value.trim();
            if (!desc) { showToast('请输入指标描述', 'error'); return; }
            
            const btn = document.getElementById('generate-metric-btn');
            btn.disabled = true;
            btn.textContent = '生成中...';
            
            try {
                const result = await apiCall('/api/metrics/generate', 'POST', { description: desc });
                if (result.success) {
                    await apiCall('/api/metrics', 'POST', result.metric);
                    await loadMetrics();
                    document.getElementById('metric-description').value = '';
                    showToast(`指标"${result.metric.name}"已添加`, 'success');
                } else {
                    showToast(result.message, 'error');
                }
            } catch (e) {
                showToast('生成失败，请检查API配置', 'error');
            } finally {
                btn.disabled = false;
                btn.textContent = '✨ 生成';
            }
        }
        
        async function addManualMetric() {
            const name = document.getElementById('manual-metric-name').value.trim();
            const desc = document.getElementById('manual-metric-desc').value.trim();
            const min = parseInt(document.getElementById('manual-metric-min').value) || 0;
            const max = parseInt(document.getElementById('manual-metric-max').value) || 100;
            
            if (!name) { showToast('请输入指标名称', 'error'); return; }
            
            await apiCall('/api/metrics', 'POST', { name, description: desc, min, max });
            await loadMetrics();
            document.getElementById('manual-metric-name').value = '';
            document.getElementById('manual-metric-desc').value = '';
            showToast(`指标"${name}"已添加`, 'success');
        }
        
        async function deleteMetric(id) {
            await apiCall('/api/metrics', 'DELETE', { id });
            await loadMetrics();
            showToast('指标已删除', 'success');
        }
        
        // 模拟控制
        function updateSimPanel() {
            const agentList = document.getElementById('sim-agent-list');
            agentList.innerHTML = state.agents.map(a => `
                <div class="sim-agent">
                    <div class="sim-agent-avatar">${(a.name || '?')[0]}</div>
                    <div style="min-width: 0; flex: 1;">
                        <div class="sim-agent-name">${a.name}</div>
                        <div class="sim-agent-goal">${(a.goal || '').slice(0, 20)}...</div>
                    </div>
                </div>
            `).join('');
            
            document.getElementById('sim-world-name').textContent = state.world.name || '-';
            document.getElementById('sim-world-background').textContent = state.world.background || '-';
            document.getElementById('sim-world-rules').textContent = state.world.rules || '-';
            
            updateSimMetrics();
        }
        
        async function updateSimMetrics() {
            const panel = document.getElementById('sim-metrics-panel');
            const list = document.getElementById('sim-metrics-list');
            const chartSection = document.getElementById('chart-section');
            
            if (state.metrics.length === 0) {
                panel.style.display = 'none';
                chartSection.style.display = 'none';
                return;
            }
            
            panel.style.display = 'block';
            chartSection.style.display = 'block';
            
            const metricData = await apiCall('/api/metrics/data');
            
            list.innerHTML = state.metrics.map(m => {
                const data = metricData[m.id] || [];
                const lastValue = data.length > 0 ? data[data.length - 1].value : '-';
                const percent = data.length > 0 ? ((lastValue - (m.min || 0)) / ((m.max || 100) - (m.min || 0)) * 100) : 0;
                
                return `
                    <div class="metric-item">
                        <div class="metric-header">
                            <span class="metric-name">${m.name}</span>
                            <span class="metric-value">${typeof lastValue === 'number' ? lastValue.toFixed(1) : lastValue}</span>
                        </div>
                        <div class="metric-bar">
                            <div class="metric-bar-fill" style="width: ${percent}%"></div>
                        </div>
                    </div>
                `;
            }).join('');
            
            updateMetricsChart(metricData);
        }
        
        function initMetricsChart() {
            const ctx = document.getElementById('metrics-chart');
            if (!ctx) return;
            
            if (metricsChart) metricsChart.destroy();
            
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: { labels: [], datasets: [] },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom', labels: { color: '#888', font: { size: 10 } } } },
                    scales: {
                        x: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                        y: { ticks: { color: '#888', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
                    }
                }
            });
        }
        
        function updateMetricsChart(metricData) {
            if (!metricsChart || state.metrics.length === 0) return;
            
            const colors = ['#64ffda', '#a78bfa', '#ffd93d', '#ff6b6b', '#6bcb77'];
            const allRounds = new Set();
            Object.values(metricData).forEach(data => data.forEach(d => allRounds.add(d.round)));
            const rounds = Array.from(allRounds).sort((a, b) => a - b);
            
            metricsChart.data.labels = rounds.map(r => `#${r}`);
            metricsChart.data.datasets = state.metrics.map((m, i) => {
                const data = metricData[m.id] || [];
                const values = rounds.map(r => { const p = data.find(d => d.round === r); return p ? p.value : null; });
                return { label: m.name, data: values, borderColor: colors[i % colors.length], backgroundColor: colors[i % colors.length] + '20', tension: 0.3, fill: false };
            });
            metricsChart.update();
        }
        
        async function toggleSimulation() {
            if (state.running) await stopSimulation();
            else await startSimulation();
        }
        
        async function startSimulation() {
            const speed = parseInt(document.getElementById('speed-slider').value);
            const result = await apiCall('/api/simulation/start', 'POST', { speed });
            if (result.error) { showToast(result.error, 'error'); return; }
            state.running = true;
            updateSimulationUI();
            startPolling();
            showToast('模拟已启动', 'success');
        }
        
        async function stopSimulation() {
            await apiCall('/api/simulation/stop', 'POST');
            state.running = false;
            updateSimulationUI();
            stopPolling();
            showToast('模拟已暂停', 'info');
        }
        
        async function stepSimulation() {
            const result = await apiCall('/api/simulation/step', 'POST');
            if (result.error) { showToast(result.error, 'error'); return; }
            await pollHistory();
        }
        
        function updateSimulationUI() {
            const startBtn = document.getElementById('start-btn');
            const statusBadge = document.getElementById('status-badge');
            if (state.running) {
                startBtn.textContent = '⏸️ 暂停';
                statusBadge.textContent = '运行中';
                statusBadge.classList.remove('status-stopped');
                statusBadge.classList.add('status-running');
            } else {
                startBtn.textContent = '▶️ 开始';
                statusBadge.textContent = '已停止';
                statusBadge.classList.remove('status-running');
                statusBadge.classList.add('status-stopped');
            }
        }
        
        function startPolling() { if (pollInterval) return; pollInterval = setInterval(pollHistory, 1000); }
        function stopPolling() { if (pollInterval) { clearInterval(pollInterval); pollInterval = null; } }
        
        async function pollHistory() {
            const status = await apiCall('/api/simulation/status');
            state.round = status.round;
            document.getElementById('round-display').textContent = status.round;
            
            const history = await apiCall(`/api/history?since=${state.historyLength}`);
            
            if (history.length > 0) {
                const logsContainer = document.getElementById('sim-logs');
                if (state.historyLength === 0) logsContainer.innerHTML = '';
                
                history.forEach(log => {
                    const entry = document.createElement('div');
                    entry.className = `log-entry ${log.error ? 'error' : ''} ${log.event ? 'event' : ''}`;
                    entry.innerHTML = `
                        <div class="log-meta">
                            <span class="log-round">#${log.round}</span>
                            <span class="log-agent">${log.agent}</span>
                            <span class="log-time">${new Date(log.timestamp).toLocaleTimeString()}</span>
                        </div>
                        ${log.event ? `<div class="log-event-tag">⚡ 事件: ${log.event}</div>` : ''}
                        <div class="log-content">${escapeHtml(log.content)}</div>
                    `;
                    logsContainer.appendChild(entry);
                });
                
                state.historyLength += history.length;
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
            
            await updateSimMetrics();
            
            if (!status.running && state.running) {
                state.running = false;
                updateSimulationUI();
                stopPolling();
            }
        }
        
        async function clearHistory() {
            if (!confirm('确定要清空所有历史记录吗？')) return;
            await apiCall('/api/history/clear', 'POST');
            state.historyLength = 0;
            state.round = 0;
            document.getElementById('round-display').textContent = '0';
            document.getElementById('sim-logs').innerHTML = '<div class="empty-logs">点击"开始"或"单步"按钮启动模拟...</div>';
            await updateSimMetrics();
            showToast('历史已清空', 'success');
        }
        
        async function injectEvent() {
            const input = document.getElementById('event-input');
            const event = input.value.trim();
            if (!event) return;
            
            const result = await apiCall('/api/event', 'POST', { event });
            if (result.success) {
                input.value = '';
                showToast('事件已注入，将在下一回合生效', 'success');
            } else {
                showToast(result.error || '事件注入失败', 'error');
            }
        }
        
        document.getElementById('speed-slider').addEventListener('input', function() {
            document.getElementById('speed-value').textContent = this.value;
        });
        
        // 导入导出
        async function exportData() {
            const data = await apiCall('/api/export');
            const json = JSON.stringify(data, null, 2);
            const blob = new Blob([json], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `socialsim-export-${new Date().toISOString().slice(0, 10)}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('数据已导出', 'success');
        }
        
        function showImportModal() { document.getElementById('import-modal').classList.add('active'); }
        function hideImportModal() { document.getElementById('import-modal').classList.remove('active'); }
        
        async function importData() {
            try {
                const json = document.getElementById('import-data').value;
                const data = JSON.parse(json);
                await apiCall('/api/import', 'POST', data);
                await loadWorld();
                await loadAgents();
                await loadMetrics();
                hideImportModal();
                showToast('数据已导入', 'success');
            } catch (e) {
                showToast('导入失败：' + e.message, 'error');
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        document.getElementById('event-input').addEventListener('keypress', e => { if (e.key === 'Enter') injectEvent(); });
        document.getElementById('metric-description').addEventListener('keypress', e => { if (e.key === 'Enter') generateMetric(); });
        
        loadWorld();
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    print('''
╔═══════════════════════════════════════════════════════════╗
║     ◈  SocialSim v2.1 - AI社会模拟平台                     ║
║                                                           ║
║     开发者：Zhengxuanjiang                                 ║
║                                                           ║
║     访问 http://localhost:5000 开始使用                    ║
╚═══════════════════════════════════════════════════════════╝
    ''')
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
