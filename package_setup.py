from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root=Path(__file__).resolve().parent
delivery=Path(r'C:\Users\Drona\Downloads\ChatGPT\Graph_Memory_Lab')
delivery.mkdir(parents=True,exist_ok=True)
evidence={name:json.loads((root/name).read_text(encoding='utf-8')) for name in
          ['runtime_verified.json','graph_verified.json','extension_verified.json','pyg_verified.json']}
assert evidence['extension_verified.json']['extension_compile_and_execution']=='PASS'
assert all(x['passed'] for x in evidence['graph_verified.json']['checks'].values())
assert 'add_one' in (root/'nsys_kernel_summary.csv').read_text(encoding='utf-8-sig')
report={'verified_date':'2026-10-23','workspace':str(root),'delivery':str(delivery),
        'environment':str(root/'.venv'),'evidence':evidence,
        'nsight_systems_cuda_kernel_trace':'PASS',
        'nsight_compute_counters':'BLOCKED: ERR_NVGPUCTRPERM; no driver setting changed',
        'controller_tests_passed':13,'optimization_benchmark_run':False,
        'simulator_installed':False,'full_project_complete':False}
(root/'setup_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
suffixes={'.py','.cu','.ps1','.md','.json','.txt','.log','.csv','.nsys-rep'}
files=[p for p in root.iterdir() if p.is_file() and (p.suffix in suffixes or p.name=='.gitignore')]
for path in files:
    shutil.copy2(path,delivery/path.name)
    assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((delivery/path.name).read_bytes()).digest()
shutil.copy2(root/'SETUP_VERIFIED.md',delivery.parent/'Graph_Memory_Lab_SETUP.md')
shutil.copy2(root/'Hardware_Control_and_Simulation.md',delivery.parent/'Graph_Memory_Lab_Hardware_and_Simulation.md')
archive=delivery.parent/'Graph_Memory_Lab_Source_and_Verified_Setup.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p,Path('Graph_Memory_Lab')/p.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'copied_files':len(files),'delivery':str(delivery),'archive':str(archive),
                  'byte_identity':'PASS','zip_integrity':'PASS'},indent=2))
