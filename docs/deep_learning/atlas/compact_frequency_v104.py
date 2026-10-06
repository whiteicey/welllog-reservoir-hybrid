import json

def annotate(root,node,edge,end):
    study=root/'results/deep_learning/compact_frequency_v104_20261001'
    if not study.exists():return
    read=lambda p:json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else {}
    ablation=read(study/'ablation_report.json')
    for mode in ('local','spectral'):
        ident='pure_dl_compact_frequency_v104_fold5/'+mode;out=root/'outputs'/ident
        state=read(out/'runtime_status.json');report=read(out/'report.json');assembled=read(study/(mode+'_assembled.json'))
        text='未启动，无成绩'
        if report:text='最终SW='+str(report['target_score'])+'，实际最佳恢复完成'
        elif 'best' in state:text='训练快照SW='+str(state['best'])+'，epoch='+str(state['epoch'])+'，非最终增益'
        elif state:text='状态='+state.get('status','未知')+'，尚无成绩'
        shown=dict(resultText=text)
        if assembled:
            row=dict(value=assembled['scores']['TOTAL'],scores=assembled['scores'],kind='fifth',fold=5,scope='原第五固定三通道，4依赖阶段，第5尚未执行',source=str((study/(mode+'_assembled.json')).relative_to(root)))
            shown=dict(primary=row,results=[row])
        node(ident,'v104 SW深度滤波 '+mode,
             '原206随机CompactSW逐点96维TCN后追加零初始滤波。local65实系数/频域33real32imag同6432新增参数，全845913。频域FFT2L零扩展后裁剪，原点分辨率，同数据/Q50/own SW/SAM.1/utility1/80-18；串行，无旧SW权重。',text,
             stage='architecture',status='recorded' if report else 'running_snapshot' if state.get('status')=='running' else 'failed' if state.get('status')=='failed' else 'prepared',
             evidence=['results/deep_learning/compact_frequency_v104_20261001/README.md'],**shown)
        edge('pure_dl_compact_v100_fold5/sw_joint',ident,'evolution','同随机完整SW骨干与机制，新增同容量零初始滤波，不加载旧训练SW')
    edge('pure_dl_compact_frequency_v104_fold5/local','pure_dl_compact_frequency_v104_fold5/spectral','comparison','预定义同参数/初始化，改变线性深度滤波感受范围')
    edge('pure_dl_spectral_sequence','pure_dl_compact_frequency_v104_fold5/spectral','comparison','旧首折瓶颈FFT8负结果，不是当前逐点SW/FFT33/零扩展/匹配卷积对照')
    if ablation and ablation['predefined_frequency_context_delta']<=0:
        end('dead:v104_SW_frequency','v104 当前逐点频域滤波无匹配收益','architecture',['pure_dl_compact_frequency_v104_fold5/spectral'],
            '最终spectral-local SW='+str(ablation['predefined_frequency_context_delta']),
            '关闭当前位置/模式数/窗口设置的频域扫描，不泛化所有长尺度结构。')
