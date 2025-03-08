import { Routes } from '@angular/router';
import {DashboardComponent} from './components/dashboard/dashboard.component';
import {PageDoesNotExistComponent} from './components/page-does-not-exist/page-does-not-exist.component';

export const routes: Routes = [
  {path: "dashboard", component: DashboardComponent},
  {path: "" , redirectTo: "/dashboard", pathMatch: "full"},
  {path: "**", component: PageDoesNotExistComponent},
];
