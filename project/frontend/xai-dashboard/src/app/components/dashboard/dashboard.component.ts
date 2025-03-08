// src/app/dashboard/dashboard.component.ts

import { Component, OnInit } from '@angular/core';
import { XaiService } from '../../services/xai.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.css'],
  standalone: true
})
export class DashboardComponent implements OnInit {
  sampleData: any = {}; // Define sample input data
  integratedGradientsResult: any;
  limeExplanation: any;
  shapPlot: any;
  decisionTreeExplanation: any;

  constructor(private xaiService: XaiService) { }

  ngOnInit(): void {
    // Initialize sample data with dummy values (update to match your model features)
    this.sampleData = {
    "BELNR": "131910",
    "WAERS": "O43",
    "BUKRS": "R07",
    "KTOSL": "N80",
    "PRCTR": "J39",
    "BSCHL": "T90",
    "HKONT": "V92",
    "DMBTR": "92445518.2981",
    "WRBTR": "59585041.1988"
  };
  }

  getIntegratedGradients(): void {
    this.xaiService.getIntegratedGradients(this.sampleData).subscribe(result => {
      console.log(result)
      this.integratedGradientsResult = result.top_attributions;
    });
  }

  getLimeExplanation(): void {
    this.xaiService.getLimeExplanation(this.sampleData).subscribe(result => {
      this.limeExplanation = result.lime_explanation;
    });
  }

  getShapExplanation(): void {
    this.xaiService.getShapExplanation(this.sampleData).subscribe(result => {
      this.shapPlot = 'data:image/png;base64,' + result.shap_plot;
    });
  }

  getDecisionTreeExplanation(): void {
    this.xaiService.getDecisionTreeExplanation(this.sampleData).subscribe(result => {
      this.decisionTreeExplanation = result.decision_tree_explanation;
    });
  }
}
