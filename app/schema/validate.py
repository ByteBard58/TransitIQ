from pydantic import BaseModel, Field
from typing import Annotated

class UserInput(BaseModel):
  koi_period : Annotated[float, Field(
    ..., gt=0, description="Orbital Period (days)",
    examples= [0.837, 2.154, 9.88, 54.3, 365.2]
  )]
  koi_time0bk : Annotated[float, Field(
    ..., gt=2_450_000, description="Transit Epoch (BJD)",
    examples=[2454833.0, 2455002.5]
  )]
  koi_depth : Annotated[float, Field(
    ..., gt = 0, le = 1_000_000,
    description="Transit Depth (ppm)",
    examples=[150.2, 892.5, 3400.0]
  )]
  koi_prad : Annotated[float, Field(
    ..., description="Planet Radius (Earth radii)",
    gt=0, examples=[0.84, 1.2, 2.5, 6.8, 14.3]
  )]
  koi_sma : Annotated[float, Field(
    ..., examples=[0.021, 0.085, 0.234],
    gt = 0, description="Semi-Major Axis (AU)"
  )]
  koi_incl : Annotated[float, Field(
    ..., description="Inclination (deg)",
    gt=0, le=90, examples=[74.5,20.1,86.4]
  )]
  koi_teq : Annotated[float, Field(
    ..., description= "Equilibrium Temperature (K)",
    gt = 0, examples=[312.5, 542.0, 876.3]
  )]
  koi_insol : Annotated[float, Field(
    ..., examples=[0.32, 1.02, 4.75, 28.6, 310.0],
    description= "Insolation Flux (Earth flux)", gt =0
  )]
  koi_impact : Annotated[float, Field(
    ..., description="Impact Parameter",
    examples=[0.02, 0.18, 0.45, 0.72, 0.95], 
    ge = 0, lt = 1
  )]
  koi_ror : Annotated[float, Field(
    ..., description="Planet/Star Radius Ratio",
    examples=[0.011, 0.028, 0.065, 0.112, 0.198],
    gt = 0, lt = 1
  )] 
  koi_srho : Annotated[float, Field(
    ..., description="Stellar Density (g/cm³)", 
    gt = 0, examples=[0.18, 0.85, 1.41, 3.72, 18.6]
  )]
  koi_dor : Annotated[float, Field(
    ..., examples=[2.3, 8.7, 21.4, 56.8, 134.2],
    gt = 1, description="Planet-Star Distance (R★)"
  )]
  koi_num_transits : Annotated[int, Field(
    ..., description="Number of Transits", 
    ge=1, examples= [1, 3, 7, 15, 42]
  )]