"""Recreate the synthetic ticket bundle without third-party dependencies."""

import argparse
import collections
import csv
from pathlib import Path
import random
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-csv", type=Path, default=Path(__file__).resolve().parents[1] / "data/samples/tickets.csv")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/samples/files")
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

def write_csv(name, columns, rows):
    with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)

random.seed(42)
with args.base_csv.open(newline="", encoding="utf-8") as handle:
    src = list(csv.DictReader(handle))

# Component banks per label: (subject, symptom) pairs + framing + context
banks = {
 "access": {
  "pfx":"A",
  "core":[
   "I cannot sign in to {sys}","my {sys} password has expired","{sys} says my account is disabled",
   "I need read access to the {res}","please revoke {sys} access for a leaver in my team",
   "MFA prompt never appears when I log in to {sys}","my SSO session keeps asking me to re-authenticate on {sys}",
   "I was added to the wrong group and cannot see the {res}","my account got locked after too many attempts on {sys}",
   "request to elevate my role to approver in {sys}","the {res} shows access denied for me",
   "authenticator app was reset and I cannot complete MFA for {sys}","new joiner needs an account created in {sys}",
   "service account credentials for {sys} need rotating","my permissions were removed after the role change and I need the {res} back",
  ],
  "sys":["the HR portal","the expense tool","Okta","the CRM","Jira","the payroll system","SharePoint","the VPN portal","Workday","the analytics workspace"],
  "res":["finance shared folder","sales dashboard","audit report","procurement site","project repository","board pack folder","customer data extract"],
 },
 "hardware": {
  "pfx":"H",
  "core":[
   "my {dev} is physically damaged","the {dev} will not power on","the {dev} overheats after an hour",
   "the {noisy} makes a clicking noise","a key on my {kb} came off","my {bat} battery swelled",
   "the {prn} in {loc} keeps jamming","the {bat} hinge snapped","the {scr} screen has dead pixels",
   "the {bat} stopped charging even with a new cable","need a replacement {dev}, the old one was dropped",
   "the {scr} display flickers on its own","spilled coffee on my {dev}",
   "the {dev} arrived broken in the box","the {dev} power adapter smells burnt","the {dev} casing is cracked","the {dev} cable is frayed","the {scr} has a vertical line across it","the {bat} USB-C port is loose",
  ],
  "dev":["laptop","monitor","docking station","keyboard","headset","desk phone","printer","webcam","mouse","label printer","badge reader"],
  "kb":["laptop","keyboard"],"bat":["laptop","tablet"],"prn":["printer","label printer"],"scr":["laptop","monitor"],"noisy":["laptop","printer","desk phone","docking station"],
  "loc":["the second floor print room","reception","the Chennai office","meeting room 4B","the warehouse"],
 },
 "network": {
  "pfx":"N",
  "core":[
   "the VPN tunnel drops every {n} minutes","wifi in {loc} is down","packet loss on video calls from {loc}",
   "cannot reach the internal file server from {loc}","DNS lookups for intranet sites time out",
   "the ethernet port in {loc} shows no link light","bandwidth is very low in {loc} since morning",
   "site-to-site link to the branch office is flapping","guest wifi captive page will not load in {loc}",
   "remote desktop session freezes over the VPN","proxy is blocking the vendor portal for everyone in {loc}",
   "latency to the cloud region spiked to over {n}00 ms","the access point in {loc} keeps rebooting",
   "wifi authenticates but gets no IP address in {loc}","video calls from {loc} keep buffering","cannot ping the gateway from {loc}",
   "VPN client shows connected but no internal sites load","network drive mapping fails every {n} minutes","speed test from {loc} shows under 1 Mbps",
  ],
  "loc":["meeting room 4B","the third floor","the Bengaluru branch","the cafeteria","the training room","my home office","the warehouse"],
  "n":["2","3","5","10","15"],
 },
 "software": {
  "pfx":"S",
  "core":[
   "{app} crashes when I open large files","{app} throws error code {code} on startup",
   "please install {app} on my machine","{app} update failed halfway","{app} is stuck on the loading screen",
   "{app} freezes when I click save","{app} plugin is incompatible after the upgrade",
   "license expired message in {app}","{app} exports corrupt PDF files","{app} sync shows a conflict on every edit",
   "{app} uses 100 percent CPU and hangs","macro stopped working in {app} after patch Tuesday",
   "need {app} downgraded to the previous version",
  ],
  "app":["Excel","Outlook","the accounting app","Teams","the design tool","Chrome","the ERP client","Visual Studio Code","Adobe Reader","the BI desktop app"],
  "code":["0x80070005","E1042","500","0xC0000142","ERR_9","1603"],
 },
}
frames=["{x}","{X}.","Hi team, {x}.","{X} - please help.","Urgent: {x}","{X}, this is blocking my work today.",
        "Since this morning {x}.","{X}. Tried restarting already.","FYI {x}, colleagues see the same.","{X}?"]

def fill(t,b):
    return re.sub(r"\{(\w+)\}", lambda m: random.choice(b[m.group(1)]), t)

def make(label, k, seen):
    b=banks[label]; out=[]
    tries=0
    while len(out)<k and tries<20000:
        tries+=1
        core=fill(random.choice(b["core"]),b)
        txt=random.choice(frames).replace("{X}",core[0].upper()+core[1:]).replace("{x}",core)
        key=core.lower()
        if key in seen: continue
        seen.add(key); out.append(txt)
    return out

seen={r["text"].lower() for r in src}
rows=[(r["id"],r["text"],r["label"]) for r in src]
for label,b in banks.items():
    for i,t in enumerate(make(label,90,seen), start=11):
        rows.append((f"{b['pfx']}{i:02d}",t,label))
write_csv("tickets_sample.csv", ["id","text","label"], rows)

# Hard / ambiguous challenge set (held out; never mixed into training)
hard=[
 ("C01","VPN login rejects my password but wifi works fine","access","Credential rejection = access, even when the VPN is the entry point"),
 ("C02","VPN connects then drops after two minutes","network","Session established then lost = transport problem"),
 ("C03","Webcam not detected after the latest driver update","software","Driver/update regression = software unless device is physically faulty"),
 ("C04","Webcam lens is cracked","hardware","Physical damage"),
 ("C05","Printer shows offline for everyone on the floor","network","Shared device unreachable for many users = connectivity"),
 ("C06","Printer is out of toner","hardware","Consumable/physical"),
 ("C07","Teams says I am not allowed to join this channel","access","Authorization decision, not an app fault"),
 ("C08","Teams keeps crashing when I share my screen","software","Application fault"),
 ("C09","SharePoint is slow only from the branch office","network","Location-dependent performance"),
 ("C10","SharePoint page shows a script error for all users","software","Application error independent of user or location"),
 ("C11","laptop wont connect to anything after i dropped it","hardware","Physical cause stated; network symptom is downstream"),
 ("C12","cant login, also the screen flickers","access","Multi-issue ticket: label the blocking issue; flag for triage split"),
 ("C13","Need admin rights to install Python","access","Privilege request, not software install"),
 ("C14","Please install Python 3.12 on my laptop","software","Install request with no permission change"),
 ("C15","Docking station gives no internet through its ethernet port","hardware","Dock port fault; same port works on the wall"),
 ("C16","Outlook cannot connect to the server","network","Default to network unless account/credential error shown; review per incident"),
 ("C17","My password works on web mail but not in the Outlook app","software","Credential is valid; client misbehaving"),
 ("C18","Badge reader at the door does not accept my card","access","Physical access entitlement unless reader is dead for all"),
 ("C19","Badge reader at the door has no lights and nobody can enter","hardware","Device dead for everyone"),
 ("C20","pls help!!! nothing works since the morning","needs_triage","Insufficient signal; model should abstain"),
]
write_csv("challenge_set.csv", ["id","text","label","rationale"], hard)

# Incidents and ticket mapping (group key for leakage-safe splits)
incidents=[
 ("INC-2026-0412","VPN concentrator capacity exhaustion","network","2026-04-12","resolved"),
 ("INC-2026-0519","Okta SSO outage after certificate rotation","access","2026-05-19","resolved"),
 ("INC-2026-0603","Faulty batch of docking stations","hardware","2026-06-03","resolved"),
 ("INC-2026-0711","Accounting app crash after patch release","software","2026-07-11","resolved"),
 ("INC-2026-0822","Third floor access point firmware bug","network","2026-08-22","resolved"),
 ("INC-2026-0905","MFA SMS provider delivery delays","access","2026-09-05","monitoring"),
]
write_csv("incidents.csv", ["incident_id","title","primary_label","opened","status"], incidents)
rules={"INC-2026-0412":("network",r"vpn"),"INC-2026-0519":("access",r"sso|okta|sign on|re-authenticate"),
       "INC-2026-0603":("hardware",r"dock"),"INC-2026-0711":("software",r"accounting app"),
       "INC-2026-0822":("network",r"third floor|access point"),"INC-2026-0905":("access",r"mfa|multi factor")}
m=[]
for ticket_id,text,label in rows:
    for inc,(lab,pat) in rules.items():
        if label==lab and re.search(pat,text,re.I): m.append((ticket_id,inc)); break
write_csv("ticket_incidents.csv", ["ticket_id","incident_id"], m)
print(dict(collections.Counter(label for _,_,label in rows)), len(rows), "mapped:",len(m))
