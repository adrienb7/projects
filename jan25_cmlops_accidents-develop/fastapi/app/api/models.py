from pydantic import BaseModel, Field



class PredicResponse(BaseModel):
    predict: int
    score: float

#jour;mois;an;hrmn;lum;agg;atm;col;lat;long;catr;circ;nbv;vosp;prof;plan;surf;infra;situ;vma;senc;catv;obs;obsm;choc;manv;motor;place;catu;sexe;an_nais;secu1;secu2;secu3;locp
#14.0;7.0;2023.0;525.0;1.0;1.0;1.0;3.0;47.166595;-0.972322;3.0;2.0;0.0;0.0;1.0;1.0;1.0;0.0;1.0;80.0;2.0;10.0;0.0;0.0;8.0;1.0;1.0;1.0;1.0;1.0;1970.0;8.0;-1.0;-1.0;0.0
class PredictAccidentsPayload(BaseModel):
    jour: float
    mois: float
    an: float
    hrmn: float
    lum: float
    agg: float
    atm: float
    col: float
    lat: float 
    long: float
    catr: float
    circ: float
    nbv: float
    vosp: float
    prof: float
    plan: float
    surf: float
    infra: float
    situ: float
    vma: float
    senc: float
    catv: float
    obs: float
    obsm: float
    choc: float
    manv: float
    motor: float
    place: float
    catu: float
    sexe: float
    an_nais: float
    secu1: float
    secu2: float
    secu3: float
    locp: float


