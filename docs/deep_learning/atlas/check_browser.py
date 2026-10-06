"""Real browser acceptance for the offline atlas; no dependencies installed."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
errors=[];checks=[]
with sync_playwright() as p:
    browser=p.chromium.launch(channel='msedge',headless=True,args=['--disable-gpu'])
    page=browser.new_page(viewport={'width':1440,'height':900},device_scale_factor=1)
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto((HERE/'index.html').as_uri())
    page.wait_for_function('window.atlas && window.atlas.getVisible().length > 0')
    checks.append({'name':'offline_load','nodes':page.evaluate('atlas.data.nodes.length'),'edges':page.evaluate('atlas.data.edges.length')})
    page.screenshot(path=str(HERE/'preview-desktop.png'))
    page.locator('.node[data-id="simple_cnn_cv"]').click()
    assert page.locator('#detail-body h2').inner_text()=='简单 CNN 基线'
    before=page.locator('#zoom').inner_text()
    page.locator('#plus').click()
    assert page.locator('#zoom').inner_text()!=before
    page.locator('#minus').click()
    page.locator('.node[data-id="simple_cnn_cv"]').focus()
    page.keyboard.press('Enter')
    checks.append({'name':'svg_click_keyboard_and_zoom','passed':True})
    page.get_by_role('button',name='列表',exact=True).click()
    page.locator('#search').fill('GR 条件')
    assert page.locator('#rows tr').count()>=1
    page.locator('[data-row="pure_dl_paired_gr_conditional80_sequence"]').click()
    assert 'GR 条件参照' in page.locator('#detail-body h2').inner_text()
    page.get_by_role('button',name='追溯这条路线',exact=True).click()
    assert len(page.evaluate('atlas.getVisible()'))>1
    page.get_by_role('button',name='图谱',exact=True).click()
    page.get_by_role('button',name='在图中定位',exact=True).click()
    assert page.locator('.node.chosen').count()==1
    checks.append({'name':'search_list_detail_lineage_locate','passed':True})
    page.locator('#search').fill('no_such_experiment_99999')
    assert page.locator('#empty').is_visible()
    page.get_by_role('button',name='清除搜索',exact=True).click()
    page.locator('[data-mode="all"]').click()
    assert len(page.evaluate('atlas.getVisible()'))==page.evaluate('atlas.data.nodes.length')
    geometry=page.evaluate('''() => {
      const g=atlas.getLayout(),m=new Map(g.nodes.map(n=>[n.id,n]));
      const backwards=g.edges.filter(e=>m.get(e.target).x<=m.get(e.source).x+m.get(e.source).w);
      const lanes=new Map();for(const n of g.nodes){if(!lanes.has(n.theme))lanes.set(n.theme,new Set());lanes.get(n.theme).add(n.y)}
      const splitThemes=[...lanes].filter(([t,ys])=>ys.size!==1).map(([t])=>t);
      const reversingPaths=[...document.querySelectorAll('.edge')].filter(p=>{
        const l=p.getTotalLength();let last=-Infinity;
        for(let i=0;i<=24;i++){const x=p.getPointAtLength(l*i/24).x;if(x<last-.01)return true;last=x}return false;
      }).length;
      return {edges:g.edges.length,backwards:backwards.length,splitThemes,reversingPaths,laneCount:lanes.size};
    }''')
    assert geometry['backwards']==0 and geometry['reversingPaths']==0 and not geometry['splitThemes']
    checks.append({'name':'all_edges_left_to_right_one_row_per_theme',**geometry})
    page.locator('#fit').click()
    page.screenshot(path=str(HERE/'preview-overview.png'))
    checks.append({'name':'all_records_and_empty','passed':True})
    page.locator('[data-mode="dead"]').click()
    assert page.locator('.node.deadend').count()>=7
    page.evaluate("atlas.select('dead:generative',true)")
    assert '82.1340' in page.locator('#detail-body').inner_text()
    page.screenshot(path=str(HERE/'preview-deadend.png'))
    checks.append({'name':'dead_end_reason_and_scores','passed':True})
    for w,h in [(1366,768),(1920,1080),(390,844)]:
        page.set_viewport_size({'width':w,'height':h})
        page.evaluate("atlas.changeMode('pair')")
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.screenshot(path=str(HERE/f'preview-{w}.png'))
        checks.append({'name':f'viewport_{w}x{h}','horizontal_overflow':False})
    page.locator('#list-button').click()
    page.locator('#search').fill('GR 条件')
    page.locator('[data-row="pure_dl_paired_gr_conditional80_sequence"]').click()
    assert page.locator('#details').is_visible()
    page.locator('#close-detail').click()
    assert not page.locator('#details').is_visible()
    checks.append({'name':'mobile_search_and_detail','passed':True})
    browser.close()
result={'passed':not errors,'page_errors':errors,'checks':checks}
(HERE/'browser-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(result,ensure_ascii=True))
assert not errors
