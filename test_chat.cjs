const assert=require('node:assert/strict'),fs=require('node:fs');
const {route}=require('./web/chat.js');
const data={runtime:JSON.parse(fs.readFileSync('web/data/runtime.json')),audit:JSON.parse(fs.readFileSync('web/data/scope_audit.json'))};
let n=0;function check(fn){fn();n++;}
check(()=>assert.equal(route('what did we build for A and B?',data).kind,'scope'));
check(()=>assert.equal(route('results',data).kind,'results'));
check(()=>assert.equal(route('help',data).kind,'suggestions'));
check(()=>assert.equal(route('replay council step 1',data).step,1));
check(()=>assert.throws(()=>route('replay council step 999',data)));
check(()=>assert.equal(route('run replay',data).kind,'local-control'));
check(()=>assert.equal(route('job status',data).kind,'local-control'));
check(()=>assert.equal(route('cancel replay',data).kind,'local-control'));
check(()=>assert.throws(()=>route('delete raw evidence',data)));
check(()=>assert.throws(()=>route('<script>alert(1)</script>',data)));
check(()=>assert.throws(()=>route('x'.repeat(513),data)));
check(()=>assert.equal(route('compare uniform_train budget 100',data).kind,'evidence'));
fs.writeFileSync('results/chat_router_tests.json',JSON.stringify({status:'PASS',checks:n},null,2));console.log(n+' router checks passed');
