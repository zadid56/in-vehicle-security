close all
clc

%%%%% data processing 1 %%%%%%%%%%%%
% fileID = fopen('attack_free_dataset.txt');
% raw = textscan(fileID,'%s %s %s %s %s %s %s %s %s %s %s');
% 
% col = numel(raw);
% row = numel(raw{1});
% 
% data = zeros(row-1,col);
% 
% for i=1:row-1
%     for j=1:col
%         if(j~=2 && j~=11) 
%             data(i,j) = hex2dec(raw{j}{i+1});
%         else
%             data(i,j) = str2double(raw{j}{i+1});
%         end
%     end
% end

%%%%% data processing 2 %%%%%%%%%%%%
% data = load('can_data.mat');
% data = data.data;
% 
% db = canDatabase('hyundai_kia_generic.dbc');
% 
% msgTT = cell(size(data,1),9);
% 
% parfor i=1:size(data,1)
%     msg = canMessage(data(i,1),false,data(i,2));
%     msg.Timestamp = data(i,11);
%     msg.Data = data(i,3:2+data(i,2));
%     msgtt = canMessageTimetable(msg);
%     msgtt2 = canMessageTimetable(msgtt,db);
%     msgtable = timetable2table(msgtt2);
%     msgarray = table2cell(msgtable);
%     msgTT{i} = msgarray;
% end

%%%%% data processing 3 %%%%%%%%%%%%
% msgCell = cell(size(msgTT,1),9);
% 
% for i=1:size(msgTT,1)
%     msgCell(i,:)=msgTT{i};
% end

% a = load('/scratch2/mdzadik/can_data_processed.mat');
% data = a.data;
% 
% msgCodes = {'EMS11','EMS12','EMS13','EMS14','EMS15','EMS16','EMS17','EMS18','EMS19','EMS20','EMS21','SPAS11','SPAS12','SAS11','CAL_SAS11','GW_Warning_PE',...
%     'TCS11','TCS12','TCS13','TPMS11','EPB11','ABS11','WHL_PUL11','WHL_SPD11','MDPS11','MDPS12','LKAS11','LKAS12','FPCM11','CGW1','CGW2','CGW3','CGW4','CGW5',...
%     'ESP11','ESP12','_4WD11','_4WD12','_4WD13','BAT11','EMS_H12','TMU11','VSM11','SCC11','SCC12','SCC13','CLU11','CLU12','CLU13','CLU14','CLU15','CLU16'};
% 
% msglen = [13,19,14,8,12,15,10,5,13,3,8,8,23,5,2,7,29,3,21,12,14,7,9,8,13,11,22,5,9,43,41,4,23,25,11,14,12,6,4,9,21,8,6,15,21,3,12,1,17,26,15,3];
% 
% datacodes = string(data(:,3));
% 
% for i=1:52
%     flags(i) = mean(strcmp(datacodes,msgCodes{i}));
% end
% 
% id = find(flags);

% idx = id(7);
% code = msgCodes{idx};
% mat = zeros(1,msglen(idx)+1);
% 
% for i=1:size(data,1)
%     temp = data{i,3};
%     disp(i);
%     clc
%     if(strcmp(code,temp)==1)
%         clc
%         temp2 = cell2mat(struct2cell(data{i,5}));
%         temp3 = [seconds(data{i,1}) temp2'];
%         mat = [mat;temp3];
%         disp(i);
%     end
% end
% 
% mat(1,:)=[];

% varNames1 = {'TQ_COR_STAT','TQI_ACOR','N','TQI11','TQFR','VS'};
% varNames2 = {'MUL_CODE','TEMP_ENG','BRAKE_ACT','TPS','PV_AV_CAN'};
% varNames3 = {'VB'};
% varNames4 = {'TQI_MIN','TQI16','TQI_TARGET','TQI_MAX'};
% varNames5 = {'SAS_ANGLE','SAS_SPEED','MSGCOUNT','CHECKSUM'};
% 
% EMS11_TT = array2timetable(EMS11(:,[6,9,10,11,12,13]),'RowTimes',seconds(EMS11(:,1)),'VariableNames',varNames1);
% EMS12_TT = array2timetable(EMS12(:,[6,7,15,18,19]),'RowTimes',seconds(EMS12(:,1)),'VariableNames',varNames2);
% EMS14_TT = array2timetable(EMS14(:,7),'RowTimes',seconds(EMS14(:,1)),'VariableNames',varNames3);
% EMS16_TT = array2timetable(EMS16(:,[2,3,4,12]),'RowTimes',seconds(EMS16(:,1)),'VariableNames',varNames4);
% SAS11_TT = array2timetable(SAS11(:,[2,3,5,6]),'RowTimes',seconds(SAS11(:,1)),'VariableNames',varNames5);
% 
% data2 = synchronize(EMS11_TT,EMS12_TT,EMS14_TT,EMS16_TT,SAS11_TT,'union','linear');
% data_T = timetable2table(data2);
% writetable(data_T,'data.csv');

% a=readtable('data.csv');

% cc = zeros(20,20);
% 
% for i=1:20
%     for j=1:20
%        temp = corrcoef(a(:,i),a(:,j));
%        cc(i,j) = temp(2);
%     end
% end

bits = [2,8,16,8,8,8,2,8,2,8,8,8,8,8,8,8,16,8,4,4];
val_min = [0,0,0,0,0,0,0,-48,0,-15.0234742,0,0,0,0,0,0,-3276.8,0,0,0];
val_max = [3,99.6094,16383.75,99.6094,99.6094,254,3,143.25,3,104.6948357,99.603,25.8984375,99.609375,99.609375,99.609375,99.609375,3276.8,1016,15,15];
sd = zeros(1,20);

for i=1:20
    b = adc([val_min(i) val_max(i)],bits(i)+1,a(:,i));
    err = b'-a(:,i);
    sd(i) = 3*std(err);
end
sd=sd';
