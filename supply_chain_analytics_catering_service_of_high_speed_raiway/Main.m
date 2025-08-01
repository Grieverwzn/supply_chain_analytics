%{ 
Copyright 2023 Xin Wu - All Rights Reserved you may use, distribute and modify this code for academatic education purpose
codes of food supply chain netwok for highspeed railway
Author: Xin Wu 
%}

function [IND,RouteFlow,PJDemandTrain,Profit,MoneyShortage,MoneySurplus,TotalOC1,TotalDC1,RouteTime1,Langrange]=Main()
tic,

%% Capacity setting of diffrenet layers of vehicles: 
% 1. human loading capacity (from station to trains)
% 2. vehicle capacity (from warehouse to stations)
% 3. warehouse capacity (stored within warehouses or distribution centers)
% 4. vehicle capacity (from suppliers to warehouses) 
Capacity=[50,1000,8000,1000];

%% Capacity setting of different layers of facilities
% 1. food storage capacity in trains
% 2. food storage capacity in railway stations 
% 3. food storage capacity in warehoues 
% 4. food storage capacity in food suppliers 
CapacityFacility=[1200,1e+50,10000,1e+50];

%% basic setting of different layers (four layers)
Alpha=[0.001,0.001,0.001,0.001];
Belta=[1.1,1.1,1.1,1.1];

%% life span of food. 
LSpan=1440;
ELT=660; % earlist lunch time 
LLT=780; % latest lunch time 
EST=960; % earlist super time
LST=1200; % latest supplier time 

%% Basic configuration
SellingPrice=60;
EatRate= -(0.6*SellingPrice)/(100-30)+(0.6*100)/(100-30); % dining rate (or demand function) with respect to prices 


SeatCapacity=1200;
service_factor = 1; % the ratio between passengers and potenial services
PurchasePrice=15;
WastePrice=5;

z=0;

%% reading data
fprintf("step 1. read data ...\n")
[StoreTime,Timetable,Terminal_Timetable,DurationStaTraMatrix,...
    TimeStaTraMatrix,DecayStaTraMatrix,FixCostStaTraMatrix,...
    VarCostStaTraMatrix,DecayWare,FixCostWare,VarCostWare,...
    WarehouseStationMatrix,TimeWarStaMatrix,DecayWarStaMatrix,...
    FixCostWarStaMatrix,VarCostWarStaMatrix,PlantWarehouseMatrix,...
    TimePlaWarMatrix,DecayPlaWarMatrix,FixCostPlaWarMatrix,...
    VarCostPlaWarMatrix,OStationMatrix,DStationMatrix] = ReadingData(LSpan);
% plants have a same meaning of suppliers. 


%% create supply chain network
fprintf("step 2. create supply chain network ...\n")
%====================create storage links in warehoues===========================
[numberWarehouse,~]=size(WarehouseStationMatrix);
[numberPlant,~]=size(PlantWarehouseMatrix);
[numberTrain,numberStation]=size(Timetable);

WarehouseWarehouseMatrix=eye(numberWarehouse,numberWarehouse); % diagnoal matrices
TimeWarWarMatrix=eye(numberWarehouse,numberWarehouse); 
DecayWarWarMatrix=eye(numberWarehouse,numberWarehouse); 
FixCostWarWarMatrix=eye(numberWarehouse,numberWarehouse); 
VarCostWarWarMatrix=eye(numberWarehouse,numberWarehouse); 
for i = 1: numberWarehouse
    TimeWarWarMatrix(:,i) = WarehouseWarehouseMatrix(:,i)*StoreTime(i);
    DecayWarWarMatrix(i,i)=WarehouseWarehouseMatrix(i,i)*DecayWare(i);
    FixCostWarWarMatrix(i,i)=WarehouseWarehouseMatrix(i,i)*FixCostWare(i);
    VarCostWarWarMatrix(i,i)=WarehouseWarehouseMatrix(i,i)*VarCostWare(i);
end 


%========================create links between stations and trains===============================
TrainStationMatrix=zeros(numberTrain,numberStation);
Timetable1=Timetable;
Panalty=1e+5; % large panalty to represent the stations cannot do food catering
for i=1:numberTrain
    for j=1:numberStation
        %if Timetable(i,j)==0 || Timetable(i,j)==max(Timetable(i,:)) % not correct
        % Two kinds of stations of a train do not need to do catering
        % service : (1) the train does not stop at the station (2) the
        % train's terminal station. 
        if Timetable(i,j)==0 || Timetable(i,j)==Terminal_Timetable(i,j)
            Timetable1(i,j)=Panalty;
        end
    end
end
% Example 
% ELT earlist lunch time 
% LLT latest lunch time 
% EST earlist super time
% LST latest supplier time 
% 0:00----1----ELT==a==LLT--2---EST==b==LST---3---0:00
% Then one whole day can be divided into 5 time ranges above
for i=1:numberTrain
        [~,index_start]=max(OStationMatrix(i,:));
        [~,index_end]=max(DStationMatrix(i,:));
        % time range  a (if original station is within lunch dinning timeslot, then cater at the original station)
        if Timetable1(i,index_start)<LLT && Timetable1(i,index_start)>=ELT
            TrainStationMatrix(i,index_start)=1;
        end
        % time range 2 (if the train has dinning service requirements)
        if Timetable1(i,index_start)<EST && Timetable1(i,index_start)>=LLT && Timetable1(i,index_end)>EST
            for kk=1:numberStation
                if Timetable1(i,kk)<EST
                     TrainStationMatrix(i,kk)=1;
                end
            end
        end            
        % time range  b (if original station is within supper dinning timeslot, then cater at the original station)
        if Timetable1(i,index_start)<LST &&Timetable1(i,index_start)>=EST
            TrainStationMatrix(i,index_start)=1;
        end
        
        % time range 3 (if the train has dinning service requirements)
        if Timetable1(i,index_start)>=LST && Timetable1(i,index_end)>ELT
            for kk=1:numberStation
                if Timetable1(i,kk)<ELT || (Timetable1(i,kk)>LST && Timetable1(i,kk)<1440)
                     TrainStationMatrix(i,kk)=1;
                end
            end
        end
        % time range 1 (if the train has dinning service requirements)
        if Timetable1(i,index_start)<ELT && Timetable1(i,index_end)>ELT
            for kk=1:numberStation
                if Timetable1(i,kk)<ELT
                     TrainStationMatrix(i,kk)=1;
                end
            end
        end
    
end

fprintf("step 3. create demand of trains ...\n")
%========================create demand of trains===============================
TrainDemand=zeros(1,numberTrain);
for i=1:numberTrain
    if sum(TrainStationMatrix(i,:))>0
        TrainDemand(i)=SeatCapacity*service_factor; % the service factor express the potential dinner time in the train 
        % the service factor could be train-dependent. For example, if a
        % train passes both lunch time slot and super time slot, the
        % service factor could be 2, here we assume that service factor = 1
        % for all trains 
    else
        TrainDemand(i) = 0;
    end 
end
TrainDemand=TrainDemand';
toc;

tic,
fprintf("step 4. route searching ...\n")
% ================route searching================================================
%Findroute(route)
% ColumnPool: column 1: train id, column 2 station id, column 3 warehouse id,
% column 4: plant/supplier id 
RouteTimePool=[];
ColumnPool=[];
for t=1:numberTrain
    if TrainDemand(t)~=0 % only train demand 
        for s=1:numberStation
            for p=1:numberPlant
                route=[];
                time=0;
                if TrainStationMatrix(t,s)==1;
                    route=[t,s];
                    time=DurationStaTraMatrix(t,s);
                    time=time+TimeStaTraMatrix(t,s);
                    currentStation=s;
                    for w=1:numberWarehouse 
                        if WarehouseStationMatrix(w,currentStation)==1
                        route=[route,w];
                        time=time+TimeWarStaMatrix(w,currentStation)+TimeWarWarMatrix(w,w);
                        currentWare=w;
                        end
                    end
                    time=time+TimePlaWarMatrix(p,currentWare);
                    if time<=LSpan && TrainStationMatrix(t,s)==1
                        route=[route,p];
                        ColumnPool=[ColumnPool;route];
                        RouteTimePool=[RouteTimePool;time];
                    end      
                end
            end
        end
    end
end

RouteTimePool1=RouteTimePool;
toc;



tic,
%===============Algorithm related parameters==============
fprintf('step 5. parameter initializing ...\n')
[numberRoute,~]=size(ColumnPool);
%records the least number of routes associated with a train
TrainLeastPathNumber=sum(TrainStationMatrix'); 
PathTrainNumber=zeros(1,numberRoute);
for t=1:numberTrain
    for r=1:numberRoute
        if ColumnPool(r,1)==t
            PathTrainNumber(r)=TrainLeastPathNumber(t); 
            % realize how many other path can also possible service the train
            % just for algorithm purpose
        end
    end
end

Condition=1;
AA=[];
BB=[];
%STEP=0.1;
%set the least number as the intial step size, assume marginal val = 1 dollar
STEPRoute=1./PathTrainNumber;
STEPRouteOrigin=STEPRoute;
%LagSTEP=1e-10*ones(1,numberRoute);
LagSTEP=1e-3*ones(1,numberRoute);
Maxiter=1000;
STEP1=9;
STEP=10;
Count=0;

% Augmented Lagrangian relaxation: 
% if the penalty function is r*(g(x))^m
% r = 0.5*CC if CC = 1 then r= 1/2 just like normal Augmented 
CC=10;%·Pay attention this is an intial penalty value that is not updated per iterations

LagSTEP1=1.1;
epsilon1=10;
epsilon2=0.25;
epsilon3=5;%numberRoute;
epsilon4=0.1;
UU=LSpan-RouteTimePool;
UU=UU./min(UU);
FirstTimeCater00=35;
FirstTimeCater=FirstTimeCater00*UU;
%===================
OriginStep = Maxiter/5;%ceil(EatRate*15);
KK1=inf;
RouteFlow1=inf*ones(1,numberRoute);
%fviter0=[];
fviter=[];
fviter1=[];
numberiter=0;
%numberiter1=0;
Langrange=zeros(1,numberRoute);
Curve=[];
RouteTimeTotal=[];

% Pen=LagSTEP;
%MU=1e+6;
%MaxProfit=0;

%%=====================initialization=================================
fprintf("step 6. path flow initializing ...\n")
% two different types of assignment schemes: 
RouteFlow=0*ones(1,numberRoute); 
% initial flow == 0 
RouteFlowNoDeadline=0*ones(1,numberRoute); 
% initial flow without considering time deadline constraints == 0 

toc;

%=====================iternations================================
fprintf('step 7. start iterations ...\n')
RouteChange=[];

while Condition==1
    tic,
    numberiter=numberiter+1;
    fprintf('iteration number =', numberiter, '...\n')
    %======================flow assignment÷=============================
    % four matrices to express flow patterns: 
        % 1. FlowStationTrainMatrix
        % 2. FlowWarehouseStationMatrix
        % 3. FlowWarehouseWarehouseMatrix
        % 4. FlowPlantWarehouseMatrix
    % Since the assignment is conducted on food supply chain network the
    % link decay should be considered, the input for assignment includes: 
        % 1. ColumnPool or route set
        % 2. ColumnPool Flow 
        % 3. four layers of decay matries 
    [FlowStationTrainMatrix,FlowWarehouseStationMatrix,...
        FlowWarehouseWarehouseMatrix, FlowPlantWarehouseMatrix]...
        =Flowassignment(ColumnPool,RouteFlow,DecayPlaWarMatrix,...
        DecayWarWarMatrix,DecayWarStaMatrix,DecayStaTraMatrix);
    
    [FlowStationTrainMatrix1,FlowWarehouseStationMatrix1,...
        FlowWarehouseWarehouseMatrix1, FlowPlantWarehouseMatrix1]...
        =Flowassignment(ColumnPool,RouteFlowNoDeadline,DecayPlaWarMatrix,...
        DecayWarWarMatrix,DecayWarStaMatrix,DecayStaTraMatrix);
    
    %=========================operational cost and discarding cost derivatives===========================
    % function Derivatives for operational cost and discard cost
    % DecayRouteMatrix should be used in generalized deadline constraints 
    [IterOperationCost,IterDecayCost,DecayRouteMatrix,DecayRoute]...
        =OperationDiscardCostDev(ColumnPool,Capacity,FixCostPlaWarMatrix,...
        VarCostPlaWarMatrix,DecayPlaWarMatrix,FlowPlantWarehouseMatrix,...
        FixCostWarWarMatrix,VarCostWarWarMatrix,DecayWarWarMatrix,...
        FlowWarehouseWarehouseMatrix,FixCostWarStaMatrix,...
        VarCostWarStaMatrix,DecayWarStaMatrix,FlowWarehouseStationMatrix,...
        FixCostStaTraMatrix,VarCostStaTraMatrix,DecayStaTraMatrix,...
        FlowStationTrainMatrix,PurchasePrice,WastePrice);
    % DecayRouteMatrix1 should not be used when we do not consider
    % generalized deadline constraints 
    [IterOperationCost1,IterDecayCost1,~,DecayRoute1]...
        =OperationDiscardCostDev(ColumnPool,Capacity,FixCostPlaWarMatrix,...
        VarCostPlaWarMatrix,DecayPlaWarMatrix,FlowPlantWarehouseMatrix1,...
        FixCostWarWarMatrix,VarCostWarWarMatrix,DecayWarWarMatrix,...
        FlowWarehouseWarehouseMatrix1,FixCostWarStaMatrix,...
        VarCostWarStaMatrix,DecayWarStaMatrix,FlowWarehouseStationMatrix1,...
        FixCostStaTraMatrix,VarCostStaTraMatrix,DecayStaTraMatrix,...
        FlowStationTrainMatrix1,PurchasePrice,WastePrice);
    
    %==============================Generalized timedeadline==================
    [GeneralizedCost]=...
        GeneralizedDeadlineCost(Langrange,ColumnPool,DecayRouteMatrix,...
        FlowPlantWarehouseMatrix, TimePlaWarMatrix,...
        FlowWarehouseWarehouseMatrix, TimeWarWarMatrix,...
        FlowWarehouseStationMatrix, TimeWarStaMatrix,...
        FlowStationTrainMatrix, TimeStaTraMatrix,...
        Alpha,Belta,CapacityFacility);

    %===========================Vendor boy model=======================================
    [ProjectDemandRoute,ProjectDemandTrain,TrainDemandRoute,VendorBoy]=...
        VendorBoyModel(EatRate,ColumnPool,DecayRoute,numberTrain,...
        numberRoute,RouteFlow,TrainDemand,SellingPrice,PurchasePrice,...
        WastePrice);
    [ProjectDemandRoute1,ProjectDemandTrain1,~,VendorBoy1]=...
        VendorBoyModel(EatRate,ColumnPool,DecayRoute1,...
        numberTrain,numberRoute,RouteFlowNoDeadline,...
        TrainDemand,SellingPrice,PurchasePrice,WastePrice);
    toc, 
    disp('calculate projective direction')
    tic,
    %================¸calculate lagrangian multiplier and augmented penalty and lead time================
    % RouteTimePool :lead time from supplier to train 
    % RouteTimePool1: lead time from supplier to train, plus the duration
    % in the train 
    % RouteTimePool2: lead time lead time from supplier to train, plus the duration
    % in the train, but do not consider time-deadline constraints.
    % LagSTEP is the penalty of augmented lagrangian relaxation. 
    % PP is the max(0,RouteTimePool1(r)-LSpan-DiracFunction(RouteFlow(r)));
    % to monitor the states of time-deadline constraints. 
    [Langrange,RouteTimePool,PP,judge,RouteTimePool1]=...
        Multiplier(RouteFlow,Langrange,LagSTEP,LSpan,DurationStaTraMatrix,...
        ColumnPool,FlowPlantWarehouseMatrix,TimePlaWarMatrix,...
        FlowWarehouseWarehouseMatrix,TimeWarWarMatrix,...
        FlowWarehouseStationMatrix,TimeWarStaMatrix,...
        FlowStationTrainMatrix,TimeStaTraMatrix,...
        Alpha,Belta,CapacityFacility);
    
    [RouteTimePool2]=LeadTime(DurationStaTraMatrix,ColumnPool,...
        FlowPlantWarehouseMatrix1,TimePlaWarMatrix,...
        FlowWarehouseWarehouseMatrix1,TimeWarWarMatrix,...
        FlowWarehouseStationMatrix1,TimeWarStaMatrix,...
        FlowStationTrainMatrix1,TimeStaTraMatrix,Alpha,Belta,CapacityFacility);
    
    for r=1:numberRoute
          Penality(r)= CC*LagSTEP(r)*PP(r)*GeneralizedCost(r);
    end
    toc, 

    tic,

    %=========================step size updates 1===================
    if numberiter==1
        PPOrigin=PP;
    end
    for r=1:numberRoute
        %if norm(PP)/norm(PPOrigin)>0.25 && norm(PPOrigin)~=0;
        if norm(PP(r))/norm(PPOrigin(r))>epsilon2 && norm(PPOrigin(r))>=epsilon3 && norm(PP(r))>=epsilon3;
            LagSTEP(r)=LagSTEP(r)*LagSTEP1;% the penalty coefficients are increasing 
        % elseif norm(PP(r))/norm(PPOrigin(r))>30 && norm(PPOrigin(r))>=epsilon3 && norm(PP(r))>=epsilon3;
        %     LagSTEP(r)=LagSTEP(r)*LagSTEP1*LagSTEP1;
        else
            LagSTEP(r)=LagSTEP(r);
        end
    end
    PPOrigin=PP;

    %==========================step size updates 2==========================
    FF=zeros(1,numberRoute);% variational inequality considering penalty and timedeadline 
    FF1=zeros(1,numberRoute); % variational inequaliy without considering time deadline 
    FF2=zeros(1,numberRoute); % the gradients withtout consider penalty


    for r=1:numberRoute
        FF(r)=VendorBoy(r)-IterOperationCost(r)-IterDecayCost(r)-GeneralizedCost(r)-Penality(r);
        FF2(r)=VendorBoy(r)-IterOperationCost(r)-IterDecayCost(r)-GeneralizedCost(r);
        FF1(r)=VendorBoy1(r)-IterOperationCost1(r)-IterDecayCost1(r);
        %STEPused=STEP;
        if numberiter==1
            % initial step size
            if FF(r)>0 && PathTrainNumber(r)>0
                STEPRoute(r)=FirstTimeCater(r)./(PathTrainNumber(r).*FF(r));
                %STEPRoute(r)=FirstTimeCater./(PathTrainNumber(r));
                STEPRouteOrigin(r)=STEPRoute(r);
            end
            if FF1(r)>0 && PathTrainNumber(r)>0
                STEPRoute1(r)=FirstTimeCater(r)./(PathTrainNumber(r).*FF1(r));
                %STEPRoute1(r)=FirstTimeCater./(PathTrainNumber(r));
                STEPRouteOrigin1(r)=STEPRoute1(r);
            end
        end
        RouteFlow(r)=max(0,RouteFlow(r)+STEPRoute(r)*FF(r));
        RouteFlowNoDeadline(r)=max(0,RouteFlowNoDeadline(r)+STEPRoute1(r)*FF1(r));% Euler
    end
    toc, 


    disp('Euler and lagrangian updates')
    tic,

    % statistics 
    %=======================without timedeadline constraints===============================
%     FlowStationTrainMatrix=zeros(numberTrain,numberStation);
%     FlowWarehouseStationMatrix=zeros(numberWarehouse,numberStation);
%     FlowWarehouseWarehouseMatrix=zeros(numberWarehouse,numberWarehouse);
%     FlowPlantWarehouseMatrix=zeros(numberPlant,numberWarehouse);

    %
    [TotalOC0,TotalDC0]=FlowOperationDisCardCost(RouteFlowNoDeadline,ColumnPool,DecayPlaWarMatrix,VarCostPlaWarMatrix,FixCostPlaWarMatrix,DecayWarWarMatrix,VarCostWarWarMatrix,FixCostWarWarMatrix,DecayWarStaMatrix,VarCostWarStaMatrix,FixCostWarStaMatrix,DecayStaTraMatrix,VarCostStaTraMatrix,FixCostStaTraMatrix,Capacity,PurchasePrice,WastePrice);
    [MoneyShortage0,MoneySurplus0]=FlowIncome(ColumnPool,DecayRoute,PurchasePrice,SellingPrice,WastePrice,RouteFlowNoDeadline,TrainDemand,EatRate);
    Profit0=MoneyShortage0+MoneySurplus0-TotalOC0-TotalDC0;
    %=======================timedeadline constraints===============================

    [numberRoute,~]=size(ColumnPool);
%     FlowStationTrainMatrix=zeros(numberTrain,numberStation);
%     FlowWarehouseStationMatrix=zeros(numberWarehouse,numberStation);
%     FlowWarehouseWarehouseMatrix=zeros(numberWarehouse,numberWarehouse);
%     FlowPlantWarehouseMatrix=zeros(numberPlant,numberWarehouse);
    [TotalOC1,TotalDC1]=FlowOperationDisCardCost(RouteFlow,ColumnPool,DecayPlaWarMatrix,VarCostPlaWarMatrix,FixCostPlaWarMatrix,DecayWarWarMatrix,VarCostWarWarMatrix,FixCostWarWarMatrix,DecayWarStaMatrix,VarCostWarStaMatrix,FixCostWarStaMatrix,DecayStaTraMatrix,VarCostStaTraMatrix,FixCostStaTraMatrix,Capacity,PurchasePrice,WastePrice);
    [MoneyShortage,MoneySurplus,PJDemandTrain]=FlowIncome(ColumnPool,DecayRoute,PurchasePrice,SellingPrice,WastePrice,RouteFlow,TrainDemand,EatRate);
    %==============================Revenue calculation======================================
    Profit=MoneyShortage+MoneySurplus-TotalOC1-TotalDC1;

    toc, 
    disp('profit and cost')
    tic,

    %=============================plotting================================================
    hold on

    fviter=[fviter Profit];
    fviter1=[fviter1 Profit0];
    drawnow

    plot(fviter,'b','LineWidth',2)%,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',5) 
    hold on
    drawnow
    plot(fviter1,'k','LineWidth',1)%,'MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',5) 
    hold on
    toc, 
    disp('plotting')
    tic,

    %================================(4) convergence=====================================
    KK=norm(RouteFlow1-RouteFlow);
    XXX=[RouteTimePool1;RouteTimePool2];
    RouteTimeTotal=[RouteTimeTotal,XXX];
    RouteChange=[RouteChange,RouteFlow'];
    Curve=[Curve,KK];
    Percent=abs((KK1/KK)-1);
    for r=1:numberRoute
        if abs(FF2(r))<=epsilon1 && norm(PP)<=epsilon3
            Count=Count+1;
            RATE=STEP1/(STEP+Count);
            STEPRoute(r)=STEPRouteOrigin(r)*RATE;
        end
    end
    for r=1:numberRoute
        if abs(FF1(r))<=epsilon1*0.01
            Count=Count+1;
            RATE=STEP1/(STEP+Count);
            STEPRoute1(r)=STEPRouteOrigin1(r)*RATE;
        end
    end
    %end
    IND=binocdf(ceil(ProjectDemandRoute),ceil(TrainDemandRoute),EatRate);
    IND1=binocdf(ceil(ProjectDemandRoute1),ceil(TrainDemandRoute),EatRate);
    %IND=poisscdf(ceil(ProjectDemandRoute),ceil(TrainDemandRoute)*EatRate);

    for r=1:numberRoute
         if 0.95>=IND(r)&& IND(r)>=0.05;
            Count=Count+1;
            RATE=STEP1/(STEP+Count);
            STEPRoute(r)=STEPRouteOrigin(r)*RATE;
        end
    end
    for r=1:numberRoute
         if 0.95>=IND1(r)&& IND1(r)>=0.05;
            Count=Count+1;
            RATE=STEP1/(STEP+Count);
            STEPRoute1(r)=STEPRouteOrigin1(r)*RATE;
        end
    end

    %==================(5) Stop test=========================================

    KK1=KK;
    if KK<epsilon4  && norm(PP)<=epsilon3;
            Condition=0;
    end
    if numberiter>=Maxiter
            Condition=0;
    end
    %ColumnPool
    FlowRouteFinal=RouteFlow1;
    RouteFlow1=RouteFlow;
    toc, 
    tic,
end
RouteChange=RouteChange';
RouteInformation=[ColumnPool,RouteFlow',RouteTimePool1'];
toc;

end
            





