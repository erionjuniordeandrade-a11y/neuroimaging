import assert from 'node:assert/strict';
import test from 'node:test';
import {analyticIdentity,responseMatchesIdentity} from '../../viewer/analytic_identity.js';

test('follow-up identity comes only from the exact displayed analytic source',()=>{
  assert.deepEqual(analyticIdentity({bankId:'cst',sourcePopulation:'bank:cst'}),{bankId:'cst'});
  assert.equal(analyticIdentity({bankId:'cst',sourcePopulation:'cut-subset:cst'}),null);
  assert.equal(analyticIdentity({bankId:'cst',sourcePopulation:'bank:fat'}),null);
  const responseHeaders=new Headers({'X-resultId':'issued-recovery'});
  assert.deepEqual(analyticIdentity({bankId:null,sourcePopulation:'recovery:filter_bank',responseHeaders}),{resultId:'issued-recovery'});
  assert.equal(analyticIdentity({bankId:null,sourcePopulation:'live-track'}),null);
});
test('an envelope response cannot attach to a different displayed source',()=>{
  const headers=new Headers({'X-bankId':'fat'});
  assert.equal(responseMatchesIdentity(headers,{bankId:'cst'}),false);
  assert.equal(responseMatchesIdentity(headers,{bankId:'fat'}),true);
  assert.equal(responseMatchesIdentity(headers,{resultId:'old-result'}),false);
});
