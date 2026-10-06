"""Explicitly launch two warm-start SW objective controls, without a queue."""
from pathlib import Path
from datetime import datetime
import ast, json, subprocess, sys
from launch_independent_sw_separation_v11 import active

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parent
CODE = ROOT/'experiments/pure_dl'
STUDY = ROOT/'results/deep_learning/independent_sw_joint_v12_20260928'
MODES = ('likelihood', 'joint_utility')
METHOD = r'''# SW联合动作效用监督v12

## 目标与假设
继续三个通道独立模型路线。本轮只训练SW，POR/PERM保持旧私有模型。
v11完全分离无收益；当前检验默认质量点与连续分布共同决定动作时，增加联合决策监督是否有增量。
预定义两组likelihood/joint_utility，均从v11shared第13轮SW73.09351278156446热启动。两组网络同为92273参数、两套48维TCN、共享平均表征；每个参数和初始预测相同。旧总体最佳SW73.62238476712884作为额外比较对象，不混淆本轮起点与总体最佳。

## 唯一实验差异
设g_i是默认概率，p_i∈R^256是条件密度概率，P_i=[g_i,(1-g_i)p_i]∈R^257。K∈R^(257×A)取原解码效用矩阵，A为原800候选与默认动作合并去重后的实际数量。预测效用U_i=P_i K∈R^A。
训练使用r_i=softmax(U_i/tau)，tau=0.05预先固定。真标签y_i对应每个动作的官方收益k_i(a)=max(0,1-|a-y_i|/(0.05*(|y_i|+1e-5)))。联合决策损失为mean_i Q_i*(1-sum_a r_i(a)k_i(a))/3。
likelihood仍为mean Q*(BCE(default)+(1-default)*两格软CE)/3。
两组分别优化L_like与L_like+0.5 L_decision。系数0.5与温度0.05不是验证搜索结果，未扫描。默认/非默认训练点都进入联合项；门控与密度都有梯度。
这是平滑的动作选择代理，不等于精确优化非连续argmax官方分数。softmax还受动作网格密度影响，需作为局限记录。推理使用完全原样的argmax效用解码，温度和联合项不进入推理，不改验证阈值。

## 控制条件
同source model_best.pt及初始SW预测；新AdamW优化器lr1e-4、weight_decay0.01、cosine80*64->1e-5，每组max80/patience18/min10，batch64、64steps，seed20261010独立窗口生成器。
所有SW网络参数均可更新；沿用v11两组各clip2的裁剪。热启动并重置优化器，不冒称v11中断续跑。原九曲线/206特征+相对深度、64训练井/16验证井、原行序/144665点评分/Q50/256密度格/800候选动作保持。
两个训练图均计算相同联合项，likelihood系数为0，使监控与计算路径接近；额外监督强度是唯一改变。前向为bfloat16AMP，联合概率与效用在float32计算。固定种子不保证所有CUDA训练逐位一致。
不加载POR/PERM预测或隐藏表征到SW损失。固定POR/PERM数组仅供验证替换诊断；无测试标签。

## 记录与验收
启动时尚未证实收益。先记录真实首轮梯度是否同时到达门控与密度，初始源预测是否一致；最终固定最佳checkpoint共同推理和状态/非默认误差分解。
候选优于本轮控制只是机制开发证据，超过旧SW才考虑替换。每次保留完整原子last_state与模型/优化器/调度器/RNG/早停计数；显式恢复，不自动重试。
最多两个GPU任务，两项均由本次明确启动。无额外超参数扫描、旧ABD恢复、补折或交付覆盖。第五折严格>80.5、完整OOF>82仍为未完成目标。

## 未验证与关闭边界
本轮不是结构比较，不重开v11拆分扫描。单种子重复开发第五折不足以确认独立新井收益；同批最优epoch选择及v11源的选择历史需要保留。负结果只约束当前表示/初始化/温度/权重/训练预算。
PyTorch softmax与BCE API仅作为实现依据；公式为本次预先设计，没有外部数据支撑预期收益。
'''
AGENTS = '''# 目标与约束
目标：固定SW结构，以两个匹配实验检验联合动作效用监督。
维护PROJECTS.md，记录目标、已完成、计划、未完成与达成后事项；每批更新。原数据/分折/评分/80-18/max2GPU不改；无其他通道共享参数、自动补折或旧交付覆盖。
本批仅likelihood/joint_utility，先记录协议后训练；初始、运行中、完成、单目标/总体分数分开说明。失败留档，不擅自续跑或扩大搜索。
'''
PROGRESS = '''# 进度
目标：检验联合默认/连续动作监督能否改善SW；保持三个通道模型独立。
已完成：核对v11完成结果，当前无运行训练；设计两个同起点、同结构的损失对照。
计划：显式训练两项，核对门控/密度实际梯度，恢复最佳模型、分解误差，并记录绝对分与匹配差值。
未完成：本批精度收益、完整OOF>82、新井泛化、续跑等价性测试。
达成后：固定真正有增量的方案再验证；不自动补折或改旧交付。
'''


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if active():
        raise RuntimeError('Training active; inspect it rather than launching two new jobs')
    trainer_path = CODE/'train_sw_joint_decision_v12.py'
    dirs = [ROOT/'outputs'/f'pure_dl_independent_sw_v12_{mode}80_fold5' for mode in MODES]
    if STUDY.exists() or trainer_path.exists() or any(p.exists() for p in dirs):
        raise RuntimeError('Study exists; do not overwrite or duplicate a launch')
    text = (CODE/'train_independent_sw_separation_v11.py').read_text(encoding='utf-8-sig')
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Trainer source changed at: '+old[:90])
        text = text.replace(old, new)
    replace('"""Standalone SW architecture experiment; no pretrained prediction enters fitting."""',
            '"""Matched warm-start SW likelihood versus joint-decision objective;fixed architecture."""')
    replace('from independent_sw_separation_v11 import Network, ARCHITECTURES, objective, training_gradient_diagnostic',
            'from sw_joint_decision_v12 import Network, MODES, DecisionObjective, verify_loss_paths')
    replace("STOP = False", "WARM = ROOT/'outputs/pure_dl_independent_sw_v11_shared80_fold5'\nSTOP = False")
    replace("ap.add_argument('--architecture', choices=ARCHITECTURES, required=True)",
            "ap.add_argument('--mode', choices=MODES, required=True)")
    replace('args = ap.parse_args()', "args = ap.parse_args()\n    args.architecture = 'shared'")
    replace('initialization = model.initialize_training_priors(data)',
            "warm = torch.load(WARM/'model_best.pt', map_location='cpu', weights_only=False)\n    model.load_state_dict(warm['state_dict'], strict=True)\n    initialization = {'kind':'warmstart_v11shared_best13', 'source':str((WARM/'model_best.pt').relative_to(ROOT)), 'source_target_score':float(warm['target_score']), 'source_sha256':hashlib.sha256((WARM/'model_best.pt').read_bytes()).hexdigest()}\n    del warm")
    replace('evaluator = Evaluator(data, master, 2, reference)',
            "evaluator = Evaluator(data, master, 2, reference)\n    criterion = DecisionObjective(evaluator.actions, evaluator.utilities, model.target_median, model.target_scale, args.mode).cuda()")
    replace("cfg = {'target': 'SW', 'architecture': args.architecture,", 
            "cfg = {'target': 'SW', 'architecture': args.architecture, 'mode':args.mode,")
    replace("'pretrained_network_used': False, 'teacher_or_prior_prediction_used': False,",
            "'pretrained_network_used': True, 'teacher_or_prior_prediction_used': False, 'warmstart_not_resume':True,")
    replace("'patience': 18, 'minimum_epochs': 10, 'lr': 6e-4, 'eta_min': 1e-5,",
            "'patience': 18, 'minimum_epochs': 10, 'lr': 1e-4, 'eta_min': 1e-5, 'utility_weight':criterion.alpha, 'temperature':criterion.temperature,")
    replace("'loss': 'assignedSW BCE(default)+nondefault soft2bin CE;fixedQ;divide3',",
            "'loss':'BCE+numericsoftCE + alpha*jointsoftactionregret;alltraininglabels;fixedQ;divide3', 'actual_action_count':len(evaluator.actions),")
    replace("[Path(__file__), Path(__file__).with_name('independent_sw_separation_v11.py')]",
            "[Path(__file__), Path(__file__).with_name('independent_sw_separation_v11.py'), Path(__file__).with_name('sw_joint_decision_v12.py')]")
    replace('optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=.01)',
            'optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)')
    replace("'scope': 'new random model with training-only prior bias, NOT old checkpoint replay'",
            "'scope':'v11shared checkpoint;fresh matched optimizer;not coldstart', 'warm_predictions_exact':bool(np.array_equal(pred,np.load(WARM/'channel_pred.npy')))")
    replace("atomic_json(out/'initial_verification.json', initial)",
            "atomic_json(out/'initial_verification.json', initial)\n        assert initial['warm_predictions_exact'] and abs(score-initialization['source_target_score'])<1e-10")
    replace('tick = time.perf_counter(); model.train(); stats = np.zeros(3)',
            'tick = time.perf_counter(); model.train(); stats = np.zeros(6)')
    replace("loss, bce, ce = objective(model, output, data['labels'][ix], data['default'][ix],\n                                      mask, data['quality'][ix])",
            "loss, terms, decision_loss = criterion(model, output, data['labels'][ix], data['default'][ix],\n                                      mask, data['quality'][ix])\n            if current == 1 and step == 0:\n                loss_paths = verify_loss_paths(decision_loss, output)\n                atomic_json(out/'decision_gradient_scope.json', loss_paths)")
    replace("'first_zero_output_step_upstream_grad_expected_zero': True", "'warm_start_head_is_nonzero': True")
    replace('stats += [float(loss.detach()), float(bce), float(ce)]',
            'stats += np.r_[float(loss.detach()), terms.cpu().numpy().astype(np.float64)]')
    replace("'training_loss': stats[0]/64, 'training_bce': stats[1]/64,\n               'training_numeric_ce': stats[2]/64, 'lr': optimizer.param_groups[0]['lr'],",
            "'training_loss':stats[0]/64, 'likelihood_loss':stats[1]/64, 'training_bce':stats[2]/64,\n               'training_numeric_ce':stats[3]/64, 'joint_decision_loss':stats[4]/64, 'soft_action_reward':stats[5]/64, 'lr':optimizer.param_groups[0]['lr'],")
    replace("diagnostic = training_gradient_diagnostic(model, data)\n    atomic_json(out/'training_gradient_diagnostic.json', diagnostic)",
            "diagnostic = json.loads((out/'decision_gradient_scope.json').read_text(encoding='utf-8'))")
    replace("report = {'fold': 5, 'screening_fold': 5, 'target': 'SW', 'architecture': args.architecture,",
            "report = {'fold':5, 'screening_fold':5, 'target':'SW', 'architecture':args.architecture, 'mode':args.mode, 'source_model':initialization,")
    replace("'trainable_parameters': cfg['trainable_parameters'], 'pretrained_network_used': False,",
            "'trainable_parameters':cfg['trainable_parameters'], 'pretrained_network_used':True, 'joint_utility_weight':criterion.alpha, 'temperature':criterion.temperature,")
    # Store mode in runtime status without changing the fixed architecture metadata.
    text = text.replace("'architecture': args.architecture, 'atlas_current_trial'", "'architecture': args.architecture, 'mode':args.mode, 'atlas_current_trial'")
    text = text.replace("'architecture': args.architecture, 'last_completed_epoch'", "'architecture': args.architecture, 'mode':args.mode, 'last_completed_epoch'")
    ast.parse(text)
    trainer_path.write_text(text, encoding='utf-8')
    STUDY.mkdir(parents=True)
    for name, body in [('AGENTS.md', AGENTS), ('PROJECTS.md', PROGRESS), ('METHOD.md', METHOD)]:
        (STUDY/name).write_text(body, encoding='utf-8')
    for out in dirs:
        out.mkdir()
        for name, body in [('AGENTS.md', AGENTS), ('PROJECTS.md', PROGRESS), ('README.md', METHOD)]:
            (out/name).write_text(body, encoding='utf-8')
    rule='\n\n## 2026-09-28 当前继续：SW联合动作效用v12\n本批likelihood/joint_utility两个匹配实验，v11shared第13轮同起点，92273参数同结构；全SW参数训练，损失分别L_like与L_like+0.5*L_joint，softaction温度0.05。原动作/分折/Q/80-18/max2不改，新lr1e-4优化器重置；POR/PERM独立不变，无旧任务、补折或交付覆盖。明确默认与连续动作都参与联合项；维护PROJECTS/README/图谱，代理损失不等同精确官方argmax。\n'
    for path in [WORK/'AGENTS.md', ROOT/'AGENTS.md', ROOT/'experiments/AGENTS.md', CODE/'AGENTS.md']:
        with path.open('a', encoding='utf-8') as f: f.write(rule)
    launches=[]
    for mode, out in zip(MODES, dirs):
        argv=[sys.executable, '-X', 'utf8', '-u', trainer_path.as_posix(), '--mode', mode, '--out', out.as_posix()]
        with (out/'train.log').open('wb') as log, (out/'train.err').open('wb') as err:
            p=subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=err, stdin=subprocess.DEVNULL, close_fds=True,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        row={'mode':mode, 'pid':p.pid, 'command':argv, 'id':out.name, 'started_at':datetime.now().astimezone().isoformat()}
        (out/'launch.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
        launches.append(row)
    (STUDY/'launches.json').write_text(json.dumps(launches, ensure_ascii=False, indent=2), encoding='utf-8')
    note='\n\n## '+datetime.now().astimezone().isoformat()+' SW联合效用v12启动\n已定义同起点损失对照并实际启动'+str([(r['mode'],r['pid']) for r in launches])+'。初始/训练/最终收益待实际记录；无其他任务或补折。研究目录results/deep_learning/independent_sw_joint_v12_20260928，完整OOF>82未完成。\n'
    for path in [WORK/'PROJECTS.md', ROOT/'PROJECTS.md']:
        with path.open('a', encoding='utf-8') as f: f.write(note)
    with (ROOT/'EXPLORATIONS.md').open('a', encoding='utf-8') as f:
        for r in launches:
            f.write('\n- [SW联合动作监督 '+r['mode']+'](outputs/'+r['id']+'/README.md)：同TCN检查点，只改变联合效用项；启动时收益未证实。\n')
    print(json.dumps({'launched':launches, 'trainer':str(trainer_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
