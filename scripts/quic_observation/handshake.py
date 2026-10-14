#!/usr/bin/env python3
"""Real maximum-workload handshake-loss experiment, separate from acceptance.

Drop one client UDP datagram containing Handshake packets, selected by ordinal.
Parse only RFC 9000 section 17's unprotected QUIC v1 packet layout. Forward every
other datagram unchanged. Socket polling bounds relay shutdown and adds no delay.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quic_profiles import verify
from quic_observation.audit import state_pair
ENV=dict(os.environ,MOIRAI_QUIC_RECOVERY_TRACE='1')

def varint(data, offset):
 length=1 << (data[offset] >> 6)
 if offset+length>len(data):raise ValueError('incomplete QUIC length')
 return int.from_bytes(data[offset:offset+length],'big') & ((1 << (length*8-2))-1),offset+length

def spaces(data):
 offset=0; result=[]
 while offset<len(data):
  first=data[offset]
  if not first&0x80:return result+['Data']
  if int.from_bytes(data[offset+1:offset+5],'big')!=1:raise ValueError('experiment expects negotiated QUIC v1')
  kind=(first>>4)&3
  if kind==3:raise ValueError('unexpected Retry')
  result.append(['Initial','0-RTT','Handshake'][kind]);offset+=5
  offset+=1+data[offset];offset+=1+data[offset]
  if kind==0:
   token,offset=varint(data,offset);offset+=token
  length,offset=varint(data,offset);offset+=length
  if offset>len(data):raise ValueError('incomplete coalesced QUIC packet')
 return result

def experiment(binary,policy,drop):
 deadline=time.monotonic()+60
 def remaining():
  duration=deadline-time.monotonic()
  if duration<=0:raise TimeoutError('original pair deadline expires')
  return duration
 out={'policy':policy,'drop_client_handshake_datagram':drop,'streams':16,'bytes':65560,'messages':128,'pair_deadline_seconds':60,'operation_deadline_seconds':5,'packet_counts':{'client':0,'server':0},'handshake_datagrams':[],'errors':[]}
 server=subprocess.Popen([str(binary),'server','127.0.0.1:0','16','128','65560',policy],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=ENV)
 ready=queue.Queue(); collected={}
 def drain():
  ready.put(server.stdout.readline());collected['stdout'],collected['stderr']=server.communicate()
 reader=threading.Thread(target=drain);reader.start(); relay=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);relay.bind(('127.0.0.1',0));relay.settimeout(.01);stop=threading.Event()
 target_host=ready.get(timeout=min(5,remaining()));host,port=json.loads(target_host)['ready'].rsplit(':',1);target=(host,int(port))
 def forward():
  client=None;ordinal=0
  try:
   while not stop.is_set():
    try:data,source=relay.recvfrom(65535)
    except socket.timeout:continue
    role='server' if source==target else 'client';out['packet_counts'][role]+=1
    if role=='client':client=source
    route=client if role=='server' else target
    kinds=spaces(data)
    if role=='client' and 'Handshake' in kinds:
     dropped=ordinal==drop
     out['handshake_datagrams'].append({'ordinal':ordinal,'spaces':kinds,'bytes':len(data),'dropped':dropped});ordinal+=1
     if dropped:continue
    relay.sendto(data,route)
  except Exception as error:out['errors'].append(repr(error))
 worker=threading.Thread(target=forward);worker.start()
 try:
  local=relay.getsockname();client=subprocess.run([str(binary),'client',f'{local[0]}:{local[1]}','16','128','65560',policy],text=True,capture_output=True,env=ENV,timeout=remaining())
  reader.join(timeout=remaining())
  if reader.is_alive():out['errors'].append('pair deadline');server.terminate();reader.join(timeout=5)
  out.update({'client_returncode':client.returncode,'server_returncode':server.returncode,'client_stdout':client.stdout,'client_stderr':client.stderr,'server_stdout':collected.get('stdout'),'server_stderr':collected.get('stderr')})
  success=client.returncode==server.returncode==0 and not out['errors']
  out['complete']=success
  if success:
   verify(json.loads(client.stdout),False,16,65560);verify(json.loads(collected['stdout']),True,16,65560)
  out['states']=state_pair(client.stderr,collected['stderr'],success)
 finally:
  stop.set();worker.join(timeout=5);relay.close()
  if worker.is_alive():raise RuntimeError('owned relay fails to stop')
  if server.poll() is None:server.kill();reader.join(timeout=5)
 return out


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--binary',required=True,type=Path)
 parser.add_argument('--output',required=True,type=Path)
 parser.add_argument('--engine-provenance',required=True,type=Path)
 parser.add_argument('--candidate-patch',type=Path)
 args=parser.parse_args()
 binary=args.binary.resolve(strict=True)
 result={'schema':1,'context':'real QUIC v1 relay; one selected client Handshake datagram drop; no shaping; supplements original acceptance',
         'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
         'experiment_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'engine_observation':json.loads(args.engine_provenance.read_text()),
         'candidate_patch_sha256':hashlib.sha256(args.candidate_patch.read_bytes()).hexdigest() if args.candidate_patch else None,
         'runs':[]}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 for repetition in range(3):
  for policy in ('cubic','newreno','bbr-experimental'):
   for drop in (0,1,2):
    record=experiment(binary,policy,drop);record['repetition']=repetition
    result['runs'].append(record);args.output.write_text(json.dumps(result)+'\n')
    print(policy,repetition,drop,record['complete'],record['client_returncode'],record['server_returncode'],flush=True)


if __name__=='__main__':
 main()
